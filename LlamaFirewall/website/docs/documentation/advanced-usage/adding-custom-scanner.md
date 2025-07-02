# Adding a Custom Scanner

LlamaFirewall is a modular security system that allows developers to create custom scanners to detect and prevent specific types of threats. In this guide, we will walk you through the process of creating a new custom scanner for LlamaFirewall.

### Step 1: Define your scanner's purpose and functionality
Before creating your custom scanner, define its purpose and functionality. What type of threat do you want to detect? What kind of input data will your scanner process? Answering these questions will help you design an effective scanner.

### Step 2: Create a New Scanner Class

Create a new Python class that inherits from the `BaseScanner` class:
```python
from llamafirewall.scanners.base_scanner import BaseScanner

class MyCustomScanner(BaseScanner):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your scanner's configuration and settings

    def scan(self, input_data):
        # Implement your scanner's logic to detect threats in the input data
        pass
```
### Step 3: Implement Your Scanner's Logic

Implement the `scan` method to detect threats in the input data. You can use various techniques such as token analysis, keyword detection, contextual analysis, and heuristics-based detection.

### Step 4: Register Your Scanner

Register your scanner with the LlamaFirewall system by adding it to the `create_scanner` function of `llamafirewall` class:
```python
elif scanner_type == ScannerType.MY_CUSTOM_SCANNER:
    return MyCustomScanner()
```

**Example Code**

Here is an example of a simple custom scanner that detects a specific keyword in the input data:
```python
from llamafirewall.scanners.base_scanner import BaseScanner

class MyKeywordScanner(BaseScanner):
    def __init__(self, config):
        super().__init__(config)
        self.keyword = "malicious"

    def scan(self, input_data):
        if self.keyword in input_data:
            return True
        return False
```
and then register it in the `create_scanner` function:
```python
...
elif scanner_type == ScannerType.MY_KEYWORD_SCANNER:
    return MyKeywordScanner()
```

And that's pretty much it! Adding a new scanner for LlamaFirewall can be done in a manageable amount of steps, with the bulk of the work being defined in the `scan` method. You will note that the complexity of your scanner will depend on the logic you want to set in place to detect threats, so be sure to follow best practices to keep a good balance between performance and effectiveness.
