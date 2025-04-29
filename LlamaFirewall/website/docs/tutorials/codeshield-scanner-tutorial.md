---
sidebar_position: 3
---

# CodeShield Scanner Tutorial

In this tutorial, we'll go through the `demo_codeshield_scanner.py` script from the [examples](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall/examples) and learn how to use the CodeShield scanner in LlamaFirewall. This scanner is designed to detect potentially insecure code from a given input text. This script demonstrates how to use the scanner to analyze two example messages: one with benign code and another with insecure code.

### Importing Dependencies

The script starts by importing the necessary dependencies:
```python
import logging
import sys

from typing import Dict, Optional

from llamafirewall import AssistantMessage, LlamaFirewall, Role, ScannerType, ScanResult
```

Similar to other examples, we import the necessary dependencies from the LlamaFirewall library. We also import the `AssistantMessage` class, which represents a message coming from an assistant.

### Defining Example Messages

The script defines two example messages using the `get_sample_messages` function:
```python
def get_sample_messages() -> Dict[str, AssistantMessage]:
    # ...
     benign_message = """Sure, here's a snippet of a math function that calculates the damped sine wave:
    ```python
    import math

    def damped_sine_wave(x):
        return math.sin(x) * math.exp(-x/5) if x >= 0 else math.log(abs(x)+1)

    print(damped_sine_wave(0))
    ```"""
    # ...
    insecure_message = """You could do that with the following code:
    ```python
    import hashlib

    def store_password(password):
        # Hashing the password using MD5 (not recommended)
        hash_object = hashlib.md5(password.encode())
        return hash_object.hexdigest())

    print(store_password("securePassword123"))
    ```"""
    # ...
```
These messages represent two different scenarios: one with benign code and another with insecure code.

### Running the CodeShield Check

```python
def run_code_shield_check(message: AssistantMessage) -> Optional[ScanResult]:
    # ...
     # Initialize LlamaFirewall with CodeShield scanner
        scanner_config = {
            Role.ASSISTANT: [ScannerType.CODE_SHIELD],
        }
        firewall = LlamaFirewall(scanner_config)
        return firewall.scan(message)
```
This function initializes LlamaFirewall with a config specifying that the CodeShield scanner should be used for assistant messages.

### Running the Scanner

The main function in the script combines the above steps to run the CodeShield scanner and then prints the results including the decision, reason and score for each scanned message.

to run it, you can use the following command:
```bash
python demo_codeshield_scanner.py
```

For which you could expect an output like the following:
```bash
 === Scan Result ===
Score: 0.0000
Decision: ScanDecision.ALLOW
Reason: default
==================
=== Scan Result ===
Score: 1.0000
Decision: ScanDecision.BLOCK
Reason: 3 unsafe function call detected:
- Use of weak hashing algorithm (CWE-CWE-327) at line 7 [Severity: warning]
- Use of a Broken or Risky Cryptographic Algorithm (CWE-CWE-327) at line 7 [Severity: warning]
- The MD5 hash function is considered insecure. Avoid using it unless explicitly needed for compatibility reasons (CWE-CWE-328) at line 7 [Severity: warning]
==================
```

As you can see, the benign message was allowed, while the insecure message was blocked. The reason for the block is that the CodeShield scanner detected three unsafe function calls in the insecure message. this is a very basic example of how the CodeShield scanner can be used to detect potentially insecure code in a given input text. Feel free to inspect other tutorials frmo the left panel and from the [examples](https://github.com/meta-llama/PurpleLlama/tree/main/LlamaFirewall/examples) folder in the repository.
