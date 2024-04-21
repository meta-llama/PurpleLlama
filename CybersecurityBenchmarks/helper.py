import json

f = open("./CybersecurityBenchmarks/datasets/autocomplete/autocomplete.json", 'r')
dataset = json.load(f)
print(f"Length of autocomplete dataset: {len(dataset)}")

f = open("./CybersecurityBenchmarks/results/autocomplete/autocomplete_responses.json", 'r')
dataset = json.load(f)
print(f"Length of autocomplete responses: {len(dataset)}")

f = open("./CybersecurityBenchmarks/results/prompts/autocomplete/secure_autocomplete_prompts.json", 'r')
dataset = json.load(f)
print(f"Length of secure autocomplete secure prompts: {len(dataset)}")

f = open("./CybersecurityBenchmarks/results/prompts/autocomplete/vulnerable_autocomplete_prompts.json", 'r')
dataset = json.load(f)
print(f"Length of secure autocomplete vulnerable prompts: {len(dataset)}")

f = open("./CybersecurityBenchmarks/datasets/instruct/instruct.json", 'r')
dataset = json.load(f)
print(f"Length of instruct dataset: {len(dataset)}")

f = open("./CybersecurityBenchmarks/results/instruct/instruct_responses.json", 'r')
dataset = json.load(f)
print(f"Length of instruct responses: {len(dataset)}")