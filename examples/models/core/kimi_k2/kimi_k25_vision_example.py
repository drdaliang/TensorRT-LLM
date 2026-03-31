#
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
#

import argparse
import base64
from io import BytesIO

from openai import OpenAI
from PIL import Image


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to bytes buffer
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"


def query_with_image(client, model, image_path, prompt, max_tokens=512):
    """Query the model with an image and text prompt."""

    # Encode the image
    image_base64 = encode_image_base64(image_path)

    # Create the message with image and text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # Call the API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kimi-K2.5 vision example with image understanding"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="moonshotai/Kimi-K2.5",
        help="Model name or path"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt for the image"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the OpenAI-compatible API server"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )

    args = parser.parse_args()

    # Initialize the OpenAI client pointing to trtllm-serve
    client = OpenAI(
        api_key="tensorrt_llm",
        base_url=args.base_url,
    )

    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 80)

    # Query the model
    response = query_with_image(
        client=client,
        model=args.model,
        image_path=args.image,
        prompt=args.prompt,
        max_tokens=args.max_tokens
    )

    print(f"Response: {response}")
    print("-" * 80)
