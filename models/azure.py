from openai import AzureOpenAI
import os
import json
from .base_model import BaseModel, ModelResponse
from .model_config import ModelConfig

class AzureModel(BaseModel):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name)
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-03-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model_name = model_name
        self.context_window = ModelConfig.MODEL_SPECS[model_name]["context_window"]

    def _execute_request(self, formatted_prompt: str, system_prompt: str) -> ModelResponse:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]

            # Get token count estimate (rough estimate: 4 chars = 1 token)
            estimated_tokens = (len(system_prompt) + len(formatted_prompt)) // 4
            self._check_token_limit(estimated_tokens)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            print(response)
            
            response_text = response.choices[0].message.content
            
            return ModelResponse(
                output=json.loads(response_text),
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