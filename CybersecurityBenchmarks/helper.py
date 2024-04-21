import json

# f = open("./CybersecurityBenchmarks/datasets/autocomplete/autocomplete.json", 'r')
# f = open("./CybersecurityBenchmarks/results/autocomplete/autocomplete_responses.json", 'r')

f = open("./CybersecurityBenchmarks/results/prompts/autocomplete/secure_autocomplete_prompts.json", 'r')

# f = open("./CybersecurityBenchmarks/results/prompts/autocomplete/vulnerable_autocomplete_prompts.json", 'r')

# f = open("./CybersecurityBenchmarks/datasets/instruct/instruct.json", 'r')

# f = open("./CybersecurityBenchmarks/results/instruct/instruct_responses.json", 'r')

dataset = json.load(f)

# fd = open("./CybersecurityBenchmarks/test.json", 'w')

# result = []
# for data1 in dataset:
#     for data2 in dataset:
#         if data1["origin_code"] == data2["origin_code"] and data1["prompt_id"] != data2["prompt_id"]:
#             result.append(data1)
#             result.append(data2)

# json.dump(result, fd, indent=2)
# fd.close()

print(len(dataset))