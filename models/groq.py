from groq import Groq
import os
import json
import re
from .base_model import BaseModel, ModelResponse
from .model_config import ModelConfig
import time

class GroqModel(BaseModel):
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        super().__init__(model_name)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name
        self.context_window = ModelConfig.MODEL_SPECS[model_name]["context_window"]
        
        # Store rate limit config
        self.rate_limit_config = ModelConfig.MODEL_SPECS[self.model_name]["rate_limit"]
        self.last_request_time = 0
        self.tokens_used_in_current_minute = 0

    def _execute_request(self, formatted_prompt: str, system_prompt: str) -> ModelResponse:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]

            estimated_tokens = (len(system_prompt) + len(formatted_prompt)) // 4
            self._check_token_limit(estimated_tokens)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            print(response_text)

            # Try to parse JSON response
            try:
                output = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    json_text = response_text.split("```json\n")[1].split("```")[0]
                    output = json.loads(json_text)
                else:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if not json_match:
                        raise
                    output = json.loads(json_match.group())
            
            return ModelResponse(
                output=output,
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
        except Exception as e:
            print(e)
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                error=str(e)
            )
