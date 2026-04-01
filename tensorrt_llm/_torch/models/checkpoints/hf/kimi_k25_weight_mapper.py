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

from typing import Dict

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    register_mapper
from tensorrt_llm._torch.models.checkpoints.hf.hf_weight_mapper import \
    HfWeightMapper


@register_mapper("HF", "KimiK25ForConditionalGeneration")
class KimiK25HfWeightMapper(HfWeightMapper):
    """
    Weight mapper for Kimi K2.5 multimodal model.

    Handles weight prefix transformations for the composite model:
    - language_model.* -> model.* (text backbone)
    - visual.* -> visual.* (vision encoder)
    """

    def preprocess_weights(self, weights: Dict[str, any]) -> Dict[str, any]:
        """
        Preprocess HuggingFace weights for K2.5.

        Removes composite model prefixes to flatten the weight hierarchy.

        Args:
            weights: Raw weights from HuggingFace checkpoint

        Returns:
            Preprocessed weights with flattened structure
        """
        processed_weights = {}

        for key, value in weights.items():
            new_key = key

            # Handle language model prefix
            if key.startswith('language_model.'):
                new_key = key.replace('language_model.', '')

                # Map layers to model.layers for DeepSeek-V3 structure
                if new_key.startswith('layers.'):
                    new_key = 'model.' + new_key

            # Keep vision weights as-is (they're handled by vision encoder)
            elif key.startswith('visual.') or key.startswith('vision_model.'):
                new_key = key

            processed_weights[new_key] = value

        return processed_weights
