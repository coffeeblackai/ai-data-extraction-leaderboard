from anthropic import Anthropic, RateLimitError
import os
import json
import re
from .base_model import BaseModel, ModelResponse
from typing import Dict, Optional
from .model_config import ModelConfig
import time

class AnthropicModel(BaseModel):
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        super().__init__(model_name)
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model_name = model_name
        self.context_window = ModelConfig.MODEL_SPECS[model_name]["context_window"]

    def _execute_request(self, formatted_prompt: str, system_prompt: str) -> ModelResponse:
        """Execute request using the Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0.5,
                system=system_prompt,
                messages=[{
                    "role": "user", 
                    "content": formatted_prompt
                }],
            )
            
            # Process response...
            response_text = response.content[0].text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON found in response")
            
            output = json.loads(json_match.group())
            
            return ModelResponse(
                output=output,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            )

        except Exception as e:
            print(f"Claude API error: {str(e)}")
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                error=str(e)
            )