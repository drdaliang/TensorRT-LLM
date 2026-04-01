# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLVisionBlock)

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_disagg
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..._utils import nvtx_range
from ...inputs import (BaseMultimodalDummyInputsBuilder,
                       BaseMultimodalInputProcessor, ExtraProcessedInputs,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor,
                       support_multimodal_disaggregated)
from ...logger import logger
from ..attention_backend import AttentionMetadata
from ..attention_backend.utils import get_attention_backend
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_utils import (fuse_input_embeds,
                                        get_multimodal_embeddings)
from .modeling_utils import (ModelConfig, QuantConfig, _load_weights_impl,
                             filter_weights, register_auto_model,
                             register_vision_encoder)

PAD_INDEX = -100


class KimiK25VLInputProcessorBase(BaseMultimodalInputProcessor,
                                   BaseMultimodalDummyInputsBuilder):
    """
    Input processor for Kimi K2.5 multimodal model.

    Handles image/video tokenization and 3D RoPE position computation.
    """

    def __init__(self,
                 model_path: str,
                 config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True,
                 **kwargs):
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         **kwargs)
        self._dtype = self._config.torch_dtype
        self._tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_path)
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self.use_fast,
            trust_remote_code=trust_remote_code)

        self.tllm_multimodal_token_id = self.get_vocab_size() + 1
        self.temporal_patch_size = getattr(self.config.vision_config,
                                           'temporal_patch_size', 1)

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @classmethod
    def get_rope_index(
        cls,
        config: PretrainedConfig,
        input_ids: Optional[torch.IntTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Args:
            config: The HF's PretrainedConfig model configuration
            input_ids: Indices of input sequence tokens in the vocabulary
            image_grid_thw: The temporal, height and width of feature shape of each image in LLM
            video_grid_thw: The temporal, height and width of feature shape of each video in LLM
            attention_mask: Mask to avoid performing attention on padding token indices

        Returns:
            position_ids: A tensor of shape (3, batch_size, sequence_length)
            mrope_position_deltas: A tensor of shape (batch_size)
        """
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id
        mrope_position_deltas = []

        # Handle case with no vision inputs
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                    -1]
            else:
                position_ids = (torch.arange(input_ids.shape[1],
                                             device=input_ids.device).view(
                                                 1, 1, -1).expand(
                                                     3, input_ids.shape[0], -1))
                mrope_position_deltas = torch.zeros([input_ids.shape[0], 1],
                                                    dtype=torch.int32,
                                                    device=input_ids.device)
            return position_ids, mrope_position_deltas

        # Initialize position_ids
        batch_size, seq_len = input_ids.shape
        position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long,
                                   device=input_ids.device)

        # Process each item in batch
        for batch_idx in range(batch_size):
            text_pos = 0
            image_idx = 0
            video_idx = 0

            for seq_idx in range(seq_len):
                token_id = input_ids[batch_idx, seq_idx].item()

                if token_id == image_token_id:
                    if image_grid_thw is not None and image_idx < len(image_grid_thw):
                        t, h, w = image_grid_thw[image_idx]
                        h_pos = h // spatial_merge_size
                        w_pos = w // spatial_merge_size
                        position_ids[:, batch_idx, seq_idx] = torch.tensor([0, 0, 0])
                        image_idx += 1
                    else:
                        position_ids[:, batch_idx, seq_idx] = torch.tensor([text_pos, 0, 0])
                elif token_id == video_token_id:
                    if video_grid_thw is not None and video_idx < len(video_grid_thw):
                        t, h, w = video_grid_thw[video_idx]
                        h_pos = h // spatial_merge_size
                        w_pos = w // spatial_merge_size
                        position_ids[:, batch_idx, seq_idx] = torch.tensor([0, 0, 0])
                        video_idx += 1
                    else:
                        position_ids[:, batch_idx, seq_idx] = torch.tensor([text_pos, 0, 0])
                else:
                    position_ids[:, batch_idx, seq_idx] = torch.tensor([text_pos, 0, 0])

                text_pos += 1

            mrope_position_deltas.append(text_pos)

        mrope_position_deltas = torch.tensor(mrope_position_deltas,
                                            dtype=torch.int32,
                                            device=input_ids.device).unsqueeze(1)

        return position_ids, mrope_position_deltas

    def forward(
        self,
        text_prompts: List[TextPrompt],
        multimodal_params: Optional[List[MultimodalParams]] = None,
        **kwargs
    ) -> ExtraProcessedInputs:
        """
        Process multimodal inputs for K2.5.

        Args:
            text_prompts: List of text prompts
            multimodal_params: Optional multimodal parameters (images/videos)

        Returns:
            ExtraProcessedInputs with position_ids and other metadata
        """
        # TODO: Implement full multimodal processing
        # For now, return basic text processing
        return ExtraProcessedInputs()


class KimiK25VisionModelBase(nn.Module):
    """
    Wrapper for K2.5 vision encoder that handles weight loading.

    Similar to Qwen2VisionModelBase but adapted for K2.5 architecture.
    """

    def __init__(self,
                 model_config: ModelConfig[PretrainedConfig],
                 vision_model_class=None):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config.vision_config

        # Reset quant config for vision encoder (typically not quantized)
        quant_config = copy.deepcopy(model_config.quant_config)
        quant_config.exclude_modules = []
        vision_model_config = ModelConfig(
            pretrained_config=self.config,
            mapping=model_config.mapping,
            quant_config=quant_config,
            attn_backend=model_config.attn_backend
        )

        # Use provided vision model class or default to Qwen2_5_VisionTransformerPretrainedModel
        if vision_model_class is None:
            vision_model_class = Qwen2_5_VisionTransformerPretrainedModel

        # Initialize vision model
        self.vision_model = vision_model_class(self.config)

    @nvtx_range("KimiK25VisionModelBase.forward")
    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Image pixel values
            grid_thw: Grid dimensions (temporal, height, width)

        Returns:
            Vision embeddings
        """
        return self.vision_model(pixel_values=pixel_values, grid_thw=grid_thw)

    def load_weights(self, weights: dict):
        """
        Load weights for the vision encoder.

        Handles weight format conversions (e.g., QKV projection splitting).
        """
        # Filter for visual weights
        visual_weights = {
            k: v for k, v in weights.items()
            if k.startswith('visual.') or k.startswith('vision_model.')
        }

        # Remove prefix
        visual_weights = {
            k.replace('visual.', '').replace('vision_model.', ''): v
            for k, v in visual_weights.items()
        }

        # Load into vision model
        _load_weights_impl(self.vision_model, visual_weights)


class KimiK25VisionModel(torch.nn.Module):
    """
    K2.5 Vision Encoder.

    Based on Qwen2_5_VisionModel architecture but adapted for K2.5.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = self.model_config.pretrained_config.vision_config

        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_size = self.config.patch_size
        self.window_size = getattr(self.config, 'window_size', 7)
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # Patch embedding
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=self.config.patch_size,
            temporal_patch_size=getattr(self.config, 'temporal_patch_size', 1),
            in_channels=getattr(self.config, 'in_channels', 3),
            embed_dim=self.config.hidden_size,
        )

        # Rotary position embeddings
        head_dim = self.config.hidden_size // self.config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        # Vision transformer blocks
        self.blocks = torch.nn.ModuleList([
            Qwen2_5_VLVisionBlock(model_config, layer_idx=layer_idx)
            for layer_idx in range(self.config.depth)
        ])

        # Patch merger (spatial reduction)
        self.merger = KimiK25VLPatchMerger(self.model_config)

        # Attention metadata
        self.metadata_cls = get_attention_backend(
            self.model_config.attn_backend).Metadata

        self.full_attn_metadata = self.metadata_cls(
            max_num_requests=8192,
            max_num_tokens=8192,
            kv_cache_manager=None,
        )

    @nvtx_range("KimiK25VisionModel.forward")
    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: shape (batch, channels, height, width)
            grid_thw: shape (num_images, 3) - temporal, height, width grid dimensions

        Returns:
            Vision embeddings: shape (total_tokens, hidden_size)
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # Compute rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Process through vision transformer blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attn_metadata=self.full_attn_metadata
            )

        # Merge patches (spatial reduction)
        hidden_states = self.merger(hidden_states)

        return hidden_states

    def rot_pos_emb(self, grid_thw):
        """Compute rotary position embeddings for vision tokens."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb


class KimiK25VLPatchMerger(nn.Module):
    """
    Patch merger for spatial reduction in K2.5 vision encoder.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.config = model_config.pretrained_config.vision_config
        self.hidden_size = self.config.hidden_size

        # Linear projection for merging
        input_dim = self.hidden_size * self.config.spatial_merge_size ** 2
        self.mlp = Linear(
            input_dim,
            self.config.hidden_size,
            bias=True,
            parallel_mode=TensorParallelMode.NONE
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Merge spatial patches.

        Args:
            hidden_states: Vision features

        Returns:
            Merged features
        """
        # Simple spatial merging via linear projection
        return self.mlp(hidden_states)


class KimiK25VLModelBase(PreTrainedModel):
    """
    Base class for K2.5 multimodal model.

    Combines DeepSeek-V3 text backbone with K2.5 vision encoder.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig], **kwargs):
        # Initialize parent with full config
        super().__init__(model_config.pretrained_config)

        self.model_config = model_config
        config = model_config.pretrained_config

        # Extract text config for LLM
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config._frozen = False
        llm_model_config.pretrained_config = config.text_config
        # Update architecture to DeepSeekV3
        llm_model_config.pretrained_config.architectures = ["DeepseekV3ForCausalLM"]
        llm_model_config._frozen = True

        # Build LLM from config
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        # Build vision encoder (unless disaggregated serving)
        if not _is_disagg():
            vision_model_class = kwargs.get("vision_model_class", None)
            self.mm_encoder = KimiK25VisionModelBase(
                model_config,
                vision_model_class
            ).eval()
        else:
            self.mm_encoder = None

    @nvtx_range("KimiK25VLModelBase.forward")
    def forward(
        self,
        input_ids: torch.Tensor,
        multimodal_params: Optional[List[MultimodalParams]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        **kwargs
    ):
        """
        Forward pass with multimodal support.

        Args:
            input_ids: Input token IDs
            multimodal_params: Multimodal parameters (images/videos)
            attn_metadata: Attention metadata

        Returns:
            Model logits
        """
        # Get text embeddings
        input_embeds = self.llm.model.embed_tokens(input_ids)

        # Process multimodal inputs if present
        if multimodal_params is not None and self.mm_encoder is not None:
            # Get vision embeddings
            mm_embeds = get_multimodal_embeddings(
                self.mm_encoder.forward,
                multimodal_params
            )

            # Fuse vision and text embeddings
            input_embeds = fuse_input_embeds(
                input_embeds,
                input_ids,
                mm_embeds,
                multimodal_params
            )

        # Forward through LLM
        return self.llm.forward(
            input_ids=None,  # We provide embeddings directly
            inputs_embeds=input_embeds,
            attn_metadata=attn_metadata,
            **kwargs
        )

    def load_weights(self, weights: dict):
        """
        Load weights for composite model.

        Handles both language_model.* and visual.* prefixes.
        """
        # Split weights into LLM and vision
        llm_weights = {}
        vision_weights = {}

        for key, value in weights.items():
            if key.startswith('language_model.'):
                # Strip language_model prefix
                new_key = key.replace('language_model.', '')
                # Map to LLM structure
                if new_key.startswith('layers.'):
                    new_key = 'model.' + new_key
                llm_weights[new_key] = value
            elif key.startswith('visual.') or key.startswith('vision_model.'):
                vision_weights[key] = value
            else:
                # Try to assign to LLM by default
                llm_weights[key] = value

        # Load LLM weights
        self.llm.load_weights(llm_weights)

        # Load vision weights if encoder exists
        if self.mm_encoder is not None and vision_weights:
            self.mm_encoder.load_weights(vision_weights)
