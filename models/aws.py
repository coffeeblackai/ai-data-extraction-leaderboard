import boto3
import json
import os
from .base_model import BaseModel, ModelResponse
from .model_config import ModelConfig

class AwsModel(BaseModel):
    def __init__(self, model_name: str = "amazon.nova-micro-v1:0"):
        super().__init__(model_name)
        self.client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.model_name = model_name
        self.context_window = ModelConfig.MODEL_SPECS[model_name]["context_window"]
        
    def _execute_request(self, formatted_prompt: str, system_prompt: str) -> ModelResponse:
        try:
            # Get token count estimate
            estimated_tokens = (len(system_prompt) + len(formatted_prompt)) // 4
            self._check_token_limit(estimated_tokens)
            
            # Determine if it's a Claude model
            is_claude_model = "anthropic.claude" in self.model_name
            
            # Prepare the request body based on model type
            request_body = self._prepare_request_body(formatted_prompt, system_prompt, is_claude_model)
            
            # Make the API call
            response = self.client.invoke_model(
                modelId=self.model_name,
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response["body"].read())
            response_text, usage = self._parse_response(response_body, is_claude_model)
            
            # Process the output
            output = self._process_output(response_text)
            
            # Return successful response - ensure output is not empty
            if output:
                return ModelResponse(
                    output=output,
                    usage=usage
                )
            else:
                raise ValueError("Empty output from model")
        except Exception as e:
            error_str = str(e).strip()

            # Only create error response if there's an actual error message
            if error_str:
                error_msg = str(e)
                return ModelResponse(
                    output={},
                    usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    error=error_msg
                )
            
            # If we have output and the error is empty/whitespace, return success response
            if output:
                return ModelResponse(
                    output=output,
                    usage=usage
                )
            
            # Fallback error case
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                error="Unknown error occurred"
            )

    def _prepare_request_body(self, formatted_prompt: str, system_prompt: str, is_claude_model: bool) -> dict:
        is_cohere_model = "cohere.command" in self.model_name
        
        if is_claude_model:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "system": [
                    {"type": "text", "text": system_prompt}
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 20
            }
        elif is_cohere_model:
            return {
                "message": formatted_prompt,
                "temperature": 0.5,
                "p": 0.9,
                "k": 20,
                "max_tokens": 1000,
                "preamble": system_prompt,
                "return_prompt": True
            }
        
        return {
            "system": [
                {"text": system_prompt}
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": formatted_prompt}
                    ]
                }
            ],
            "inferenceConfig": {
                "top_p": 0.9,
                "top_k": 20,
                "temperature": 0.5,
                "max_tokens": 1000
            }
        }

    def _parse_response(self, response_body: dict, is_claude_model: bool) -> tuple[str, dict]:
        is_cohere_model = "cohere.command" in self.model_name
        
        if is_claude_model:
            response_text = response_body["content"][0]["text"]
            usage = {
                "input_tokens": response_body["usage"]["input_tokens"],
                "output_tokens": response_body["usage"]["output_tokens"],
                "total_tokens": response_body["usage"]["input_tokens"] + response_body["usage"]["output_tokens"]
            }
        elif is_cohere_model:
            response_text = response_body["text"]
            usage = {
                "input_tokens": response_body.get("usage", {}).get("input_tokens", 0),
                "output_tokens": response_body.get("usage", {}).get("output_tokens", 0),
                "total_tokens": response_body.get("usage", {}).get("total_tokens", 0)
            }
        else:
            response_text = response_body["output"]["message"]["content"][0]["text"]
            usage = {
                "input_tokens": response_body["usage"].get("inputTokens", 0),
                "output_tokens": response_body["usage"].get("outputTokens", 0),
                "total_tokens": response_body["usage"].get("totalTokens", 0)
            }

        # If any token counts are 0, estimate them based on text length
        if usage["input_tokens"] == 0 or usage["output_tokens"] == 0:
            # Estimate ~4 characters per token as a rough approximation
            if is_claude_model:
                input_text = response_body.get("input", "")
            elif is_cohere_model:
                input_text = response_body.get("prompt", "")
            else:
                input_text = response_body.get("input", {}).get("message", {}).get("content", [{}])[0].get("text", "")
            
            estimated_input_tokens = len(input_text) // 4 if usage["input_tokens"] == 0 else usage["input_tokens"]
            estimated_output_tokens = len(response_text) // 4 if usage["output_tokens"] == 0 else usage["output_tokens"]
            
            usage = {
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "total_tokens": estimated_input_tokens + estimated_output_tokens
            }

        return response_text, usage

    def _process_output(self, response_text: str) -> dict:
        try:
            # First attempt: direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Second attempt: extract from markdown code block
            if "```json" in response_text:
                try:
                    json_text = response_text.split("```json\n")[1].split("```")[0].strip()
                    return json.loads(json_text)
                except (IndexError, json.JSONDecodeError):
                    pass
            # If both parsing attempts fail, return the raw text
            return {"text": response_text}