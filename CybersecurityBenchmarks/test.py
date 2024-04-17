import json

fr = open("./CybersecurityBenchmarks/results/autocomplete/autocomplete_responses.json", 'r')

fd = open("./CybersecurityBenchmarks/datasets/autocomplete/autocomplete.json", 'r')

responses = json.load(fr)
dataset = json.load(fd)

count = 0
f = open("./CybersecurityBenchmarks/results/prompts.json", "a")
filtered_data = []
for i in range(0, len(responses)):
    if responses[i]['icd_result'] != 0:
        count += 1
        for data in dataset:
            if responses[i]["origin_code"] == data["origin_code"]:
                filtered_data.append(data)

json.dump(filtered_data, f, indent=2)
f.close()

stat = 0
models = ["gpt-3.5-turbo"]
languages = ["c", "cpp", "csharp", "java", "javascript", "php", "python", "rust"]
fs = open("./CybersecurityBenchmarks/results/autocomplete/autocomplete_stat.json", 'r')
stats = json.load(fs)

for model in models:
    for language in languages:
        stat += stats[model][language]["vulnerable_suggestion_count"]

# print("insecure autocomplete count from stats:")
# print(stat)
# print("insecure prompt count from response:")
# print(count)