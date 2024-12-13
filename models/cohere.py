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
            print(f"Making Cohere API request with model {self.model_name}...")
            response = self.client.chat(
                model=self.model_name,
                message=full_prompt,
                temperature=0.5,
                chat_history=[],
                max_tokens=1000,
                prompt_truncation='AUTO'
            )
            print("Received response from Cohere API")
            
            # Get message output from response
            print("Extracting response text...")
            response_text = response.text
            
            # If response_text is already valid JSON, use it directly
            print("Attempting to parse response as JSON...")
            try:
                output = json.loads(response_text)
                print("Successfully parsed response as JSON")
            except json.JSONDecodeError:
                # If not JSON, return the raw text in a dictionary
                print("Response was not valid JSON, wrapping in text field")
                output = {"text": response_text}
            
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
