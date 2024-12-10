import cohere
import os
import json
import re
from .base_model import BaseModel, ModelResponse
from typing import Dict, Optional
from .model_config import ModelConfig
import time

class CohereModel(BaseModel):
    def __init__(self, model_name: str = "command"):
        super().__init__(model_name)
        self.client = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))
        self.model_name = model_name
        self.context_window = ModelConfig.MODEL_SPECS[model_name]["context_window"]
        
        # Store rate limit config 
        self.rate_limit_config = ModelConfig.MODEL_SPECS[self.model_name]["rate_limit"]
        self.last_request_time = 0
        self.tokens_used_in_current_minute = 0

    def _execute_request(self, formatted_prompt: str, system_prompt: str) -> ModelResponse:
        """Execute request using the Cohere API"""
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{formatted_prompt}"
            
            response = self.client.chat(
                model=self.model_name,
                message=full_prompt,
                temperature=0.5,
                chat_history=[],
                max_tokens=1000,
                prompt_truncation='AUTO'
            )
            # Get message output from response
            response_text = response.text
            
            # Try multiple approaches to parse JSON from response text
            output = None
            
            # First try direct JSON parsing
            try:
                output = json.loads(response_text)
            except json.JSONDecodeError:
                # Try different JSON extraction methods
                json_patterns = [
                    r'```json\n(.*?)\n```',  # JSON in code blocks with json tag
                    r'```(.*?)```',          # JSON in any code blocks
                    r'\{(?:[^{}]|(?R))*\}',  # Nested JSON objects
                    r'\{.*\}'                # Simple JSON objects
                ]
                
                for pattern in json_patterns:
                    try:
                        matches = re.findall(pattern, response_text, re.DOTALL)
                        if matches:
                            # Try each match until we find valid JSON
                            for match in matches:
                                try:
                                    output = json.loads(match.strip())
                                    break
                                except json.JSONDecodeError:
                                    continue
                            if output:
                                break
                    except Exception:
                        continue
                
                if not output:
                    print(output)
                    raise ValueError("No valid JSON found in response")

            # Get usage data from response
            billed_units = response.meta.billed_units
            usage = {
                "input_tokens": billed_units.input_tokens,
                "output_tokens": billed_units.output_tokens,
                "total_tokens": billed_units.input_tokens + billed_units.output_tokens
            }
            
            return ModelResponse(
                output=output,
                usage=usage
            )
        except Exception as e:
            print(f"Cohere API error: {str(e)}")
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                error=str(e)
            )
