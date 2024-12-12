import google.generativeai as genai
import os
import json
from .base_model import BaseModel, ModelResponse
from .model_config import ModelConfig

class GoogleModel(BaseModel):
    def __init__(self, model_name: str = "gemini-1.5-flash-002"):
        super().__init__(model_name)
        # Initialize logging before creating the model
        self.model = genai.GenerativeModel(model_name)
        self.context_window = ModelConfig.MODEL_SPECS[model_name]["context_window"]
        
    def _execute_request(self, formatted_prompt: str, system_prompt: str) -> ModelResponse:
        try:
            # Get token count estimate
            estimated_tokens = (len(system_prompt) + len(formatted_prompt)) // 4
            self._check_token_limit(estimated_tokens)
            
            # Create a chat object and combine system and user prompts
            # Combine system prompt and user prompt
            combined_prompt = f"{system_prompt}\n\n{formatted_prompt}"

            chat_session = self.model.start_chat(history=[])
            response = chat_session.send_message(
                combined_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.5,
                    top_p=0.9,
                    top_k=20,
                    max_output_tokens=8192
                )
            )
            # Try to parse JSON response
            try:
                # First try to parse the text directly from the response
                if hasattr(response, 'text'):
                    output = json.loads(response.text)
                else:
                    # If no direct text attribute, get it from parts
                    text = response.candidates[0].content.parts[0].text
                    output = json.loads(text)
            except json.JSONDecodeError:
                # Keep existing JSON extraction fallback logic
                if "```json" in str(response):
                    json_text = str(response).split("```json\n")[1].split("```")[0]
                    output = json.loads(json_text)
                else:
                    # Try to find JSON-like content between curly braces
                    import re
                    json_match = re.search(r'\{.*\}', str(response), re.DOTALL)
                    if not json_match:
                        raise
                    output = json.loads(json_match.group())
            
            # Get usage metrics from the response
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
            
            return ModelResponse(
                output=output,
                usage=usage
            )
        except Exception as e:
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                error=str(e)
            )