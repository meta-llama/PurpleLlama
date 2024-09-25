# Running Llama Guard 3-1B-INT-4 ExecuTorch Model


## Running Executorch (.pte) Model in a Python runner on Torchchat

**Step 1:** Download the .pte model and model params (see the [Llama CLI Reference](https://github.com/meta-llama/llama-stack/blob/main/docs/cli_reference.md#step-1-get-the-models) for model download instructions)

**Step 2:** Install [torchchat](https://github.com/pytorch/torchchat)

**Step 3:** Run the following command

```
 python torchchat.py generate llama3-1b-guard
 --pte-path <path_to_local_hf_repo>/llama_guard_3_1b_pruned_xnnpack.pte --params-path <path_to_local_hf_repo>/params.json --prompt "$(cat <path_to_local_hf_repo>/example-prompt.txt)"

```


## Running the Model using ExecuTorch

**Step 1:** Load the Executorch model:
Prerequisite: [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup.html#quick-setup-colab-jupyter-notebook-prototype)

```
from executorch.extension.pybindings.portable_lib import _load_for_executorch
et_model = _load_for_executorch(model_file) # model_file is the path to ET model
```

**Step 2:** Tokenize the input text (using Llama Guard 3-1B tokenizer)

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "meta-llama/Llama-Guard-3-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


prompt = "tell me a joke"
token_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt")
```

**Step 3:** Generate first token from the model

The token list is passed to the ExecuTorch model to produce the first token in the output. The model uses KV (key-value) to optimize inference performance, so we pass a parameter to specify the position of the tokens in the KV cache along with the list of tokens. The starting position of the KV cache is 0. Here is an example of how to generate the first token:

```
pos = 0  # position in KV cache
out_tokens = []  # contains the output tokens
input = (
   token_ids,
   torch.tensor([pos], dtype=torch.long),
)
logits = et_model.forward(input)
out_token = logits[0].argmax(-1).item()  # first_token
print(out_token)  # 19193 is safe, 39257 is unsafe
out_tokens.append(out_token)
```

**Step 4:** Generate additional tokens

In order to generate the next set of tokens, aditional inference can be run until a stop token is reached or the maximum number of desired tokens are generated (e.g., no more than 15). For each step, we feed the model the output token from the previous step and we set the Kv cache positions to start from the next position. For example, if the number of input tokens is 5 then we set the position for the next inference to 5 (0-based).

```
max_seq_len = 15
new_pos = len(token_ids) # New position in the KV cache (After the first n tokens)


# Loop until the stop token is reached or the maximum sequence length is exceeded

while out_token != 128009 and len(out_tokens) < max_seq_len:
   new_input = (
       torch.tensor([[out_token]], dtype=torch.long),
       torch.tensor([new_pos], dtype=torch.long),
   )

   new_logits = et_model.forward(new_input)
   out_token = new_logits[0].argmax(-1).item()
   out_tokens.append(out_token)
   new_pos += 1
print(out_tokens)
```

**Step 5:** Decode the output tokens to text

```
output_text = tokenizer.decode(out_tokens)
```
