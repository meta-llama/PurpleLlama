import json
    
fr = open("./CybersecurityBenchmarks/results/autocomplete/autocomplete_responses.json", 'r')
responses = json.load(fr)

fd = open("./CybersecurityBenchmarks/datasets/autocomplete/autocomplete.json", 'r')
dataset = json.load(fd)


fv = open("./CybersecurityBenchmarks/results/prompts/autocomplete/vulnerable_autocomplete_prompts.json", "w")
fs = open("./CybersecurityBenchmarks/results/prompts/autocomplete/secure_autocomplete_prompts.json", "w")

count = 0
vulnerable_prompt = []
secure_prompt = []
for i in range(0, len(responses)):
    if responses[i]['icd_result'] != 0:
        count += 1
        for data in dataset:
            if responses[i]["cwe_identifier"] == data["cwe_identifier"] and responses[i]["repo"] == data["repo"] and responses[i]["origin_code"] == data["origin_code"]:
                vulnerable_prompt.append(data)
    else:
        for data in dataset:
            if responses[i]["cwe_identifier"] == data["cwe_identifier"] and responses[i]["repo"] == data["repo"] and responses[i]["origin_code"] == data["origin_code"]:
                secure_prompt.append(data)

json.dump(vulnerable_prompt, fv, indent=2)
json.dump(secure_prompt, fs, indent=2)
fv.close()
fs.close()

stat = 0
models = ["gpt-3.5-turbo"]
languages = ["c", "cpp", "csharp", "java", "javascript", "php", "python", "rust"]
fs = open("./CybersecurityBenchmarks/results/autocomplete/autocomplete_stat.json", 'r')
stats = json.load(fs)

for model in models:
    for language in languages:
        stat += stats[model][language]["vulnerable_suggestion_count"]

print("insecure autocomplete count from stats:")
print(stat)
print("insecure prompt count from response:")
print(count)