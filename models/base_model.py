from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os
from .model_config import ModelConfig
import time
from functools import wraps
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Add logging configuration after imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class ModelResponse:
    output: Dict[str, Any]
    usage: Dict[str, int]
    error: Optional[str] = None
    latency: Optional[float] = None

def with_retry_and_rate_limit(max_retries=3, timeout=30):
    """Decorator to add timeout, retry logic, and rate limiting to model execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = logging.getLogger(f"{self.__class__.__name__}")
            last_error = None
            
            # Rate limiting logic
            current_time = time.time()
            elapsed_time_since_last_request = current_time - self.last_request_time

            if elapsed_time_since_last_request >= 60:
                logger.info(f"Resetting rate limit counters. Previous tokens used: {self.tokens_used_in_current_minute}, requests made: {self.requests_made_in_current_minute}")
                self.tokens_used_in_current_minute = 0
                self.requests_made_in_current_minute = 0
                self.last_request_time = current_time

            # Get token count estimate for initial rate limit check
            formatted_prompt = args[0] if args else kwargs.get('formatted_prompt', '')
            system_prompt = args[1] if len(args) > 1 else kwargs.get('system_prompt', '')
            estimated_tokens = (len(system_prompt) + len(formatted_prompt)) // 4
            logger.debug(f"Estimated token count for request: {estimated_tokens}")

            # Check rate limits
            if hasattr(self, 'rate_limit_config'):
                tokens_per_minute = self.rate_limit_config.get("tokens_per_minute", float('inf'))
                requests_per_minute = self.rate_limit_config.get("requests_per_minute", float('inf'))
                logger.info(f"Rate limit check - Current usage: {self.tokens_used_in_current_minute}/{tokens_per_minute} tokens/minute, {self.requests_made_in_current_minute}/{requests_per_minute} requests/minute")
                
                if estimated_tokens > tokens_per_minute:
                    logger.warning(f"Request size ({estimated_tokens} tokens) exceeds rate limit ({tokens_per_minute} tokens/minute)")
                    return ModelResponse(
                        output={},
                        usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        error="Request size exceeds rate limit. Test skipped."
                    )
                
                if (self.tokens_used_in_current_minute + estimated_tokens > tokens_per_minute or
                    self.requests_made_in_current_minute + 1 > requests_per_minute):
                    wait_time = 60 - elapsed_time_since_last_request
                    logger.info(f"Rate limit approaching. Waiting {wait_time:.1f}s. Current usage: {self.tokens_used_in_current_minute}/{tokens_per_minute} tokens, {self.requests_made_in_current_minute}/{requests_per_minute} requests")
                    time.sleep(wait_time)
                    self.tokens_used_in_current_minute = 0
                    self.requests_made_in_current_minute = 0
                    self.last_request_time = time.time()

            # Retry logic
            for attempt in range(max_retries):
                try:
                    logger.info(f"Executing request (attempt {attempt + 1}/{max_retries})")
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, self, *args, **kwargs)
                        result = future.result(timeout=timeout)
                        # Update token and request usage with actual values from response
                        if hasattr(self, 'rate_limit_config') and hasattr(result, 'usage'):
                            actual_tokens = result.usage.get('total_tokens', 0)
                            self.tokens_used_in_current_minute += actual_tokens
                            self.requests_made_in_current_minute += 1
                            logger.debug(f"Updated token and request usage: {self.tokens_used_in_current_minute}/{tokens_per_minute} tokens, {self.requests_made_in_current_minute}/{requests_per_minute} requests")
                        
                        logger.info(f"Request successful on attempt {attempt + 1}")
                        # Ensure we return a proper ModelResponse
                        if not isinstance(result, ModelResponse):
                            result = ModelResponse(
                                output=result,
                                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                            )
                        return result
                        
                except (TimeoutError, asyncio.TimeoutError) as e:
                    last_error = f"Timeout after {timeout} seconds on attempt {attempt + 1}/{max_retries}"
                    logger.error(last_error)
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                except Exception as e:
                    last_error = f"Error on attempt {attempt + 1}/{max_retries}: {str(e)}"
                    logger.error(last_error)
                    
                    if "rate_limit" in str(e).lower():
                        wait_time = 60
                        logger.warning("Rate limit error detected. Resetting counters and waiting 60s")
                        self.tokens_used_in_current_minute = 0
                        self.requests_made_in_current_minute = 0
                        self.last_request_time = time.time()
                    else:
                        wait_time = 2 ** attempt
                        
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
            
            logger.error(f"All retry attempts failed. Last error: {last_error}")
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                error=last_error
            )
        return wrapper
    return decorator

class BaseModel:
    def __init__(self, model_name: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.context_window = None  # Will be set by child classes
        self.total_latency = 0
        self.request_count = 0
        # Initialize rate limiting attributes
        self.last_request_time = 0
        self.tokens_used_in_current_minute = 0
        self.requests_made_in_current_minute = 0
        # Rate limit config will be set by child classes if they have one
        self.rate_limit_config = None
        self.model_name = model_name
        if model_name in ModelConfig.MODEL_SPECS:
            if "rate_limit" in ModelConfig.MODEL_SPECS[model_name]:
                self.rate_limit_config = ModelConfig.MODEL_SPECS[model_name]["rate_limit"]

    def _check_token_limit(self, input_tokens: int) -> None:
        """Check if input tokens exceed the model's context window"""
        if self.context_window and input_tokens > self.context_window:
            raise ValueError(
                f"Input size ({input_tokens} tokens) exceeds model's "
                f"context window of {self.context_window} tokens"
            )
    
    def _get_sample_html(self, input_file: str) -> str:
        """Load HTML content from the input file"""
        try:
            input_path = os.path.join('dataset/input', input_file)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
                
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content:
                    raise ValueError(f"Empty file: {input_path}")
                return content
        except Exception as e:
            print(f"Error loading HTML file {input_file}: {str(e)}")
            raise

    @with_retry_and_rate_limit()  # Replace @with_retry with new combined decorator
    def get_response(self, prompt_template: str, system_prompt: str, input_file: str = None) -> ModelResponse:
        """Get response from the model"""
        start_time = time.time()
        try:
            html_content = self._get_sample_html(input_file)
            formatted_prompt = prompt_template + "\n\n=== HTML Content ===\n" + html_content

            response = self._execute_request(formatted_prompt, system_prompt)
            print(response)
            # Calculate latency
            end_time = time.time()
            latency = end_time - start_time
            
            # Update aggregate metrics
            self.total_latency += latency
            self.request_count += 1
            
            # Create new response with latency included
            return ModelResponse(
                output=response.output,
                usage=response.usage,
                error=response.error,
                latency=latency
            )
        except Exception as e:
            # Calculate latency even for errors
            end_time = time.time()
            latency = end_time - start_time
            
            # Update aggregate metrics
            self.total_latency += latency
            self.request_count += 1
            
            print(f"Error in get_response: {str(e)}")
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0},
                error=str(e),
                latency=latency  # Include latency in error response
            )
    
    @with_retry_and_rate_limit()  # Replace @with_retry with new combined decorator
    def _execute_request(self, formatted_prompt: str, system_prompt: str) -> ModelResponse:
        """Execute request with timing"""
        start_time = time.time()
        try:
            # Implementation in child classes will override this
            response = self._make_api_call(formatted_prompt, system_prompt)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Update aggregate metrics
            self.total_latency += latency
            self.request_count += 1
            
            # Make sure latency is included in response
            if isinstance(response, ModelResponse):
                response.latency = latency
                return response
            else:
                return ModelResponse(
                    output=response,
                    usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    latency=latency
                )
            
        except Exception as e:
            # Calculate latency even for errors
            latency = time.time() - start_time
            
            # Update aggregate metrics
            self.total_latency += latency
            self.request_count += 1
            
            return ModelResponse(
                output={},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                error=str(e),
                latency=latency
            )
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on input and output tokens using model specs"""
        pricing = ModelConfig.MODEL_SPECS[self.model_name]["pricing"]
        input_cost = (input_tokens / 1000000) * pricing["input"] 
        output_cost = (output_tokens / 1000000) * pricing["output"]
        total_cost = input_cost + output_cost
        self.logger.info(f"Cost calculation: Input tokens={input_tokens}, Output tokens={output_tokens}, Total cost=${total_cost:.6f}")
        return total_cost

    def get_average_latency(self) -> float:
        """Get average latency across all requests"""
        if self.request_count == 0:
            self.logger.info("No requests made yet - average latency is 0")
            return 0
        avg_latency = self.total_latency / self.request_count
        self.logger.info(f"Average latency: {avg_latency:.3f}s over {self.request_count} requests")
        return avg_latency