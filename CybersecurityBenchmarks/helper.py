import json

f = open("./CybersecurityBenchmarks/results/autocomplete/autocomplete_responses.json", 'r')

dataset = json.load(f)

print(len(dataset))