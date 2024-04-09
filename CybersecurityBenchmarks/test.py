import json

f = open("./CybersecurityBenchmarks/datasets/autocomplete/autocomplete.json", 'r')

data = json.load(f)

print(len(data))