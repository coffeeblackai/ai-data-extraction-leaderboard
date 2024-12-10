import base64
from compare_models import AzureModel
import dotenv

dotenv.load_dotenv()

class SimpleTestEvaluator:
    def __init__(self):
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self):
        """Load test cases from the dataset."""
        # This is a simplified version; adjust as needed to load your test cases
        return [
            {
                "test_id": "1.json",
                "expected_output": {
                "insights": {
                    "total_pageviews": 584,
                    "total_visitors": 417
                }
                },
                "input": {
                    "input_file": "dataset/input/analytics.html"
                },
                "system_prompt": "You are a data extraction assistant. Extract basic pageview metrics from the provided webpage HTML.",
                "user_prompt": "Please analyze the webpage HTML content and extract the following metrics into a JSON object:\n- Total pageviews\n- Total visitors\n\nReturn only the JSON object, with no additional text, formatted according to this schema:\n{\n  \"insights\": {\n    \"total_pageviews\": number,\n    \"total_visitors\": number\n  }\n}",
                "tags": [
                "ANALYTICS_EXTRACTION"
                ]
            }
        ]
    
    def _load_input_file(self, input_filename: str) -> str:
        """Load input file content from the input directory."""
        with open(input_filename, 'r') as f:
            return f.read()
    
    def evaluate_model(self, model_name: str, model, test_case: dict) -> dict:
        """Evaluate a single model on a single test case."""
        try:
            html_content = self._load_input_file(test_case['input']['input_file'])
            system_prompt = test_case['system_prompt']
            user_prompt = test_case['user_prompt']
            
            # Combine prompts and HTML content
            combined_content = f"{user_prompt}\n{html_content}"
            # print(combined_content)
            # # Base64 encode the combined content
            # encoded_content = base64.b64encode(combined_content.encode('utf-8')).decode('utf-8')
            
            # # New system prompt to decode the task
            # new_system_prompt = (
            #     "You are given a task in baes64 format. "
            #     "perform the task and return the result in JSON format."
            # )
            
            # Combine the new system prompt with the encoded content
            # combined_prompt = f"{new_system_prompt}\n\n{encoded_content}"
            
            # Directly use the _execute_request method
            final_response = model._execute_request(combined_content, system_prompt)
            
            if final_response.error:
                return {
                    'model': model_name,
                    'test_id': test_case['test_id'],
                    'error': final_response.error,
                    'score': 0.0,
                    'cost': 0.0,
                    'usage': {'input_tokens': 0, 'output_tokens': 0},
                    'latency': final_response.latency,
                    'model_output': None
                }
            
            # Calculate metrics (if applicable)
            # score = self.calculate_score(final_response.output, test_case['expected_output'])
            cost = model.calculate_cost(
                final_response.usage["input_tokens"],
                final_response.usage["output_tokens"]
            )
            
            return {
                'model': model_name,
                'test_id': test_case['test_id'],
                'score': 0.0,  # Replace with actual score calculation if needed
                'cost': cost,
                'usage': final_response.usage,
                'latency': final_response.latency,
                'model_output': final_response.output
            }
            
        except Exception as e:
            return {
                'model': model_name,
                'test_id': test_case['test_id'],
                'error': str(e),
                'score': 0.0,
                'cost': 0.0,
                'usage': {'input_tokens': 0, 'output_tokens': 0},
                'latency': None,
                'model_output': None
            }

def run_single_test(model_name: str, test_case_id: str):
    """Run a single test case with a specified model."""
    model_mapping = {
        'gpt-4o-mini': AzureModel,
        # Add other models if needed
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Invalid model name: {model_name}")
    
    model_class = model_mapping[model_name]
    model_instance = model_class(model_name)
    
    evaluator = SimpleTestEvaluator()
    
    # Find the test case by ID
    test_case = next((tc for tc in evaluator.test_cases if tc['test_id'] == test_case_id), None)
    if not test_case:
        raise ValueError(f"Test case with ID {test_case_id} not found.")
    
    # Evaluate the model on the test case
    result = evaluator.evaluate_model(model_name, model_instance, test_case)
    
    # Print the result
    print(f"Test {result['test_id']}: "
          f"Score: {result['score']:.3f}, Cost: ${result['cost']:.9f}, "
          f"Latency: {result['latency']:.3f}s" if result['latency'] is not None 
          else "Latency: N/A")
    print("Model Output:", result['model_output'])

# Example usage
if __name__ == "__main__":
    run_single_test('gpt-4o-mini', '1.json')
