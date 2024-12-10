import json

# Load the JSON data from the file
with open('published/12-08-2024-results.json', 'r') as file:
    data = json.load(file)

# Initialize counters for total tokens
total_tokens_gpt_4o_mini = 0
total_tokens_gemini_flash = 0

# Iterate over each test result
for test in data:
    model = test.get('model')
    total_tokens = test.get('usage', {}).get('total_tokens', 0)
    
    # Aggregate tokens based on the model
    if model == "gpt-4o-mini":
        total_tokens_gpt_4o_mini += total_tokens
    elif model == "gemini-1.5-flash-002":
        total_tokens_gemini_flash += total_tokens

# Print the results
print(f"Total tokens for gpt-4o-mini: {total_tokens_gpt_4o_mini}")
print(f"Total tokens for gemini-1.5-flash-002: {total_tokens_gemini_flash}")
