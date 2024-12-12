import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any
from models.azure import AzureModel
from models.anthropic import AnthropicModel
from models.google import GoogleModel
from models.aws import AwsModel
from models.groq import GroqModel
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import glob
from bs4 import BeautifulSoup
import math
from models.base_model import ModelResponse
from tqdm import tqdm
import argparse
from models.model_config import ModelConfig
from models.cohere import CohereModel

# Load environment variables
load_dotenv()

class JsonSimilarityScorer:
    """Evaluates similarity between two JSON objects using schema and semantic comparison"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the scorer with a sentence transformer model.
        
        Args:
            model_name: The HuggingFace model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {}
        
    def get_schema_similarity(self, json1: Dict[str, Any], json2: Dict[str, Any]) -> float:
        """Compare similarity of JSON structure/keys"""
        keys1 = set(self._get_all_keys(json1))
        keys2 = set(self._get_all_keys(json2))
        
        if not keys1 and not keys2:
            return 1.0
            
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union)
    
    def get_value_similarity(self, json1: Dict[str, Any], json2: Dict[str, Any]) -> float:
        """Compare semantic similarity of JSON values"""
        values1 = [str(v) for v in self._get_all_values(json1)]
        values2 = [str(v) for v in self._get_all_values(json2)]
        
        if not values1 or not values2:
            return 0.0
            
        # Get embeddings
        emb1 = self.model.encode(values1, convert_to_numpy=True)
        emb2 = self.model.encode(values2, convert_to_numpy=True)
        
        # Average the embeddings for each JSON
        avg_emb1 = np.mean(emb1, axis=0)
        avg_emb2 = np.mean(emb2, axis=0)
        
        # Compute cosine similarity
        similarity = np.dot(avg_emb1, avg_emb2) / (
            np.linalg.norm(avg_emb1) * np.linalg.norm(avg_emb2)
        )
        
        return float(similarity)
    
    def get_similarity_score(self, 
                           json1: Dict[str, Any], 
                           json2: Dict[str, Any],
                           schema_weight: float = 0.4,
                           value_weight: float = 0.6) -> float:
        """
        Get overall similarity score combining schema and value similarity.
        
        Args:
            json1: First JSON object
            json2: Second JSON object
            schema_weight: Weight for schema similarity (0-1)
            value_weight: Weight for value similarity (0-1)
            
        Returns:
            Float between 0-1 indicating similarity
        """
        schema_score = self.get_schema_similarity(json1, json2)
        value_score = self.get_value_similarity(json1, json2)
        
        return (schema_weight * schema_score) + (value_weight * value_score)
    
    def _get_all_keys(self, obj: Any, prefix: str = '') -> List[str]:
        """Recursively get all keys from nested JSON"""
        keys = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                full_key = f"{prefix}.{k}" if prefix else k
                keys.append(full_key)
                keys.extend(self._get_all_keys(v, full_key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                keys.extend(self._get_all_keys(item, f"{prefix}[{i}]"))
        return keys
    
    def _get_all_values(self, obj: Any) -> List[str]:
        """Recursively get all values from nested JSON"""
        values = []
        if isinstance(obj, dict):
            for v in obj.values():
                values.extend(self._get_all_values(v))
        elif isinstance(obj, list):
            for item in obj:
                values.extend(self._get_all_values(item))
        elif obj is not None:
            values.append(str(obj))
        return values

def chunk_html(html_content: str, model_context_window: int, base_tokens: int) -> List[str]:
    """Split HTML content into chunks based on the model's context window."""
    
    # More conservative token estimation
    response_reserve = 2000  # Increase reserve tokens for model response
    safety_margin = 4000    # Add safety margin for token estimation inaccuracy
    available_tokens = model_context_window - base_tokens - response_reserve - safety_margin
    
    # More conservative chars-to-tokens ratio (3 chars per token instead of 4)
    max_chunk_chars = available_tokens * 3
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    def finalize_current_chunk():
        if current_chunk:
            chunk_content = ''.join(current_chunk)
            # Use more conservative token estimate for verification
            chunk_tokens = len(chunk_content) // 3
            if chunk_tokens <= available_tokens:
                chunks.append(chunk_content)
            else:
                # Split into smaller chunks if needed
                for i in range(0, len(chunk_content), max_chunk_chars):
                    part = chunk_content[i:i+max_chunk_chars]
                    if len(part) // 3 <= available_tokens:
                        chunks.append(part)
            current_chunk.clear()
            return 0
        return 0

    def add_element_to_chunk(element_str: str):
        nonlocal current_chunk, current_size
        element_tokens = len(element_str) // 3  # More conservative estimate
        
        if element_tokens > available_tokens:
            # Element itself is too large, split it
            for i in range(0, len(element_str), max_chunk_chars):
                part = element_str[i:i+max_chunk_chars]
                part_tokens = len(part) // 3
                if current_size + part_tokens > available_tokens:
                    current_size = finalize_current_chunk()
                current_chunk.append(part)
                current_size += part_tokens
        else:
            # Check if adding this element would exceed the limit
            if current_size + element_tokens > available_tokens:
                current_size = finalize_current_chunk()
            current_chunk.append(element_str)
            current_size += element_tokens

    # Process major elements
    for element in soup.find_all(['table', 'tr', 'div', 'p']):
        add_element_to_chunk(str(element))
    
    # Finalize any remaining content
    if current_chunk:
        finalize_current_chunk()

    return chunks

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text (more conservative estimate)"""
    # Use more conservative ratio of 3 chars per token
    return len(text) // 3

def create_continuation_prompt(chunk: str, previous_result: dict, original_prompt: str) -> str:
    """Create a prompt for continuing analysis with a new chunk"""
    return f"""Continue analyzing the webpage HTML content based on the original task:
{original_prompt}

Previous partial results:
{json.dumps(previous_result, indent=2)}

Additional HTML content to analyze:
{chunk}

Please update the previous results with any new information found in this chunk while maintaining the same JSON structure."""

class TestEvaluator:
    def __init__(self):
        self.scorer = JsonSimilarityScorer()
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load all test cases from easy, medium, and hard directories"""
        test_cases = []
        test_dirs = ['dataset/easy', 'dataset/medium', 'dataset/hard']
        
        for dir_path in test_dirs:
            if not os.path.exists(dir_path):
                continue
                
            json_files = glob.glob(os.path.join(dir_path, '*.json'))
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    test_case = json.load(f)
                    # Add metadata about test difficulty
                    test_case['difficulty'] = os.path.basename(dir_path)
                    test_case['test_id'] = os.path.basename(json_file)
                    test_cases.append(test_case)
                    
        return test_cases
        
    def _load_input_file(self, input_filename: str) -> str:
        """Load input file content from the input directory"""
        input_path = os.path.join('dataset/input', input_filename)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        with open(input_path, 'r') as f:
            return f.read()
            
    def evaluate_model(self, model_name: str, model, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model on a single test case"""
        try:
            print("File: ", test_case['input']['input_file'])
            html_content = self._load_input_file(test_case['input']['input_file'])
            system_prompt = test_case['system_prompt']
            user_prompt = test_case['user_prompt']
            
            # Calculate base token usage
            base_tokens = estimate_tokens(system_prompt + user_prompt)
            total_estimated_tokens = estimate_tokens(html_content) + base_tokens
            
            if total_estimated_tokens > model.context_window:
                # Content needs chunking
                chunks = chunk_html(html_content, model.context_window, base_tokens)
                
                # Process first chunk with original prompt
                first_chunk = chunks[0]
                formatted_prompt = user_prompt + "\n\n=== HTML Content ===\n" + first_chunk
                response = model._execute_request(formatted_prompt, system_prompt)
                
                if response.error:
                    raise Exception(f"Error processing first chunk: {response.error}")
                
                accumulated_result = response.output
                accumulated_usage = response.usage
                accumulated_latency = response.latency or 0
                
                # Process remaining chunks
                for chunk in tqdm(chunks[1:], desc=f"Processing chunks for {model_name}", leave=False):
                    continuation_prompt = create_continuation_prompt(
                        chunk, 
                        accumulated_result,
                        user_prompt
                    )
                    chunk_response = model._execute_request(continuation_prompt, system_prompt)
                    
                    if chunk_response.error:
                        if "Rate limit exceeded" in chunk_response.error:
                            tqdm.write(f"Rate limit exceeded for {model_name} on test {test_case['test_id']}")
                            return {
                                'model': model_name,
                                'test_id': test_case['test_id'],
                                'difficulty': test_case['difficulty'],
                                'tags': test_case.get('tags', []),
                                'error': "Rate limit exceeded",
                                'score': 0.0,
                                'cost': model.calculate_cost(accumulated_usage["input_tokens"], accumulated_usage["output_tokens"]),
                                'usage': accumulated_usage,
                                'latency': accumulated_latency,
                                'test_case': test_case,
                                'model_output': accumulated_result
                            }
                        else:
                            raise Exception(f"Error in chunk processing: {chunk_response.error}")
                    
                    # Merge results
                    accumulated_result = self._merge_results(accumulated_result, chunk_response.output)
                    accumulated_usage = {
                        "input_tokens": accumulated_usage["input_tokens"] + chunk_response.usage["input_tokens"],
                        "output_tokens": accumulated_usage["output_tokens"] + chunk_response.usage["output_tokens"],
                        "total_tokens": accumulated_usage["total_tokens"] + chunk_response.usage["total_tokens"]
                    }
                    
                    accumulated_latency += chunk_response.latency or 0
                
                final_response = ModelResponse(
                    output=accumulated_result,
                    usage=accumulated_usage,
                    latency=accumulated_latency
                )
            else:
                # Content fits in context window
                final_response = model.get_response(user_prompt, system_prompt, test_case['input']['input_file'])
            
            if final_response.error:
                return {
                    'model': model_name,
                    'test_id': test_case['test_id'],
                    'difficulty': test_case['difficulty'],
                    'tags': test_case.get('tags', []),
                    'error': final_response.error,
                    'score': 0.0,
                    'cost': 0.0,
                    'usage': {'input_tokens': 0, 'output_tokens': 0},
                    'latency': final_response.latency,
                    'test_case': test_case,
                    'model_output': None
                }
            
            # Calculate metrics
            score = self.scorer.get_similarity_score(
                final_response.output,
                test_case['expected_output']
            )
            
            cost = model.calculate_cost(
                final_response.usage["input_tokens"],
                final_response.usage["output_tokens"]
            )
            
            return {
                'model': model_name,
                'test_id': test_case['test_id'],
                'difficulty': test_case['difficulty'],
                'tags': test_case.get('tags', []),
                'score': score,
                'cost': cost,
                'usage': final_response.usage,
                'latency': final_response.latency,
                'test_case': test_case,
                'model_output': final_response.output
            }
            
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {e}")
            return {
                'model': model_name,
                'test_id': test_case['test_id'],
                'difficulty': test_case['difficulty'],
                'tags': test_case.get('tags', []),
                'error': str(e),
                'score': 0.0,
                'cost': 0.0,
                'usage': {'input_tokens': 0, 'output_tokens': 0},
                'latency': None,
                'test_case': test_case,
                'model_output': None
            }
    
    def _merge_results(self, previous_result: dict, new_result: dict) -> dict:
        """Merge two result dictionaries, combining lists and updating values"""
        merged = previous_result.copy()
        
        for key, value in new_result.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list):
                # Combine lists, avoiding duplicates
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict):
                merged[key] = self._merge_results(merged[key], value)
            else:
                # For simple values, prefer the new value
                merged[key] = value
        
        return merged

def compare_models(test_suite: str = 'all', model: str = 'all', output_mode: str = 'cli'):
    model_mapping = {
        'gpt-4o-mini': AzureModel,
        # 'claude-3-opus-20240229': AnthropicModel,
        'claude-3-5-sonnet-20241022': AnthropicModel,
        'claude-3-5-haiku-20241022': AnthropicModel,
        'gemini-1.5-flash-002': GoogleModel,
        'gemini-1.5-pro': GoogleModel,
        "gemini-2.0-flash-exp": GoogleModel,
        'amazon.nova-micro-v1:0': AwsModel,
        'amazon.nova-lite-v1:0': AwsModel,
        'amazon.nova-pro-v1:0': AwsModel,
        'arn:aws:bedrock:us-east-1:502675305491:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0': AwsModel,
        'arn:aws:bedrock:us-east-1:502675305491:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0': AwsModel,
        'llama-3.1-8b-instant': GroqModel,
        'command-nightly': CohereModel,
        'command-r': CohereModel,
        'command-r-plus': CohereModel
    }
    models = {}
    if model == 'all':
        for model_name in ModelConfig.MODEL_SPECS.keys():
            model_class = model_mapping[model_name]
            models[model_name] = model_class(model_name)
    else:
        # Check if the input is a family name
        if model in ModelConfig.get_all_families():
            family_models = ModelConfig.get_models_by_family(model)
            for model_name in family_models:
                model_class = model_mapping[model_name]
                models[model_name] = model_class(model_name)
        # Otherwise treat as individual model name
        elif model in model_mapping:
            model_class = model_mapping[model]
            models[model] = model_class(model)
        else:
            raise ValueError(f"Invalid model or family name: {model}")

    evaluator = TestEvaluator()
    
    if test_suite != 'all':
        evaluator.test_cases = [tc for tc in evaluator.test_cases if tc['difficulty'] == test_suite]
    
    all_results = []
    
    # Create progress bar for models
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Add progress bar for test cases
        for test_case in tqdm(evaluator.test_cases, desc=f"Running tests for {model_name}"):
            result = evaluator.evaluate_model(model_name, model, test_case)
            all_results.append(result)
            
            # Update with inline progress
            tqdm.write(f"Test {result['test_id']} ({result['difficulty']}): "
                      f"Score: {result['score']:.3f}, Cost: ${result['cost']:.9f}, "
                      f"Latency: {result['latency']:.3f}s" if result['latency'] is not None 
                      else "Latency: N/A")
    
    if output_mode == 'file':
        # Save all results to a JSON file
        with open('model_comparison_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\nModel comparison results saved to 'model_comparison_results.json'")
    else:
        # Aggregate and print results to CLI
        print("\n=== Overall Results ===")
        for model_name in models.keys():
            model_results = [r for r in all_results if r['model'] == model_name]
            avg_score = np.mean([r['score'] for r in model_results])
            total_cost = sum(r['cost'] for r in model_results)
            avg_latency = np.mean([r['latency'] for r in model_results if r['latency'] is not None])
            
            print(f"\n{model_name}:")
            print(f"Average Score: {avg_score:.3f}")
            print(f"Total Cost: ${total_cost:.6f}")
            print(f"Average Latency: {avg_latency:.3f}s")
            
            # Print scores by difficulty
            for difficulty in ['easy', 'medium', 'hard']:
                difficulty_results = [r for r in model_results if r['difficulty'] == difficulty]
                if difficulty_results:
                    avg_difficulty_score = np.mean([r['score'] for r in difficulty_results])
                    avg_difficulty_latency = np.mean([r['latency'] for r in difficulty_results if r['latency'] is not None])
                    print(f"{difficulty.capitalize()} Score: {avg_difficulty_score:.3f}")
                    print(f"{difficulty.capitalize()} Latency: {avg_difficulty_latency:.3f}s")

if __name__ == "__main__":
    all_models = list(ModelConfig.MODEL_SPECS.keys())
    all_families = ModelConfig.get_all_families()
    
    parser = argparse.ArgumentParser(description='Compare AI models on test cases.')
    parser.add_argument('--test-suite', choices=['all', 'easy', 'medium', 'hard'], default='all',
                        help='Specify the test suite to run (default: all)')
    parser.add_argument('--model', choices=['all'] + all_models + all_families, default='all',
                        help='Specify the model or family to run (default: all)')
    parser.add_argument('--output', choices=['cli', 'file'], default='cli',
                        help='Specify the output mode (default: cli)')
    args = parser.parse_args()
    
    compare_models(args.test_suite.lower(), args.model.lower(), args.output.lower()) 