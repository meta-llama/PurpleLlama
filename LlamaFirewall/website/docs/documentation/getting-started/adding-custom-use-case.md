# Adding a Custom Use Case

### Introduction to Use Cases

In the context of this framework, a use case refers to a specific scenario where
the framework is intended to be used. Each use case may require different
scanning layers and scanners to be configured in order to effectively detect and
prevent potential security threats.

### Predefined Use Cases

LlamaFirewall comes with two predefined use cases that cater to specific
application scenarios. The `CHAT_BOT` use case which focuses in protecting
against prompt injection attacks. This is achieved by scanning for
`PROMPT_INJECTION` attacks for both the `USER` and `SYSTEM` roles:

```python
UseCase.CHAT_BOT: {
    Role.USER: [ScannerType.PROMPT_INJECTION],
    Role.SYSTEM: [ScannerType.PROMPT_INJECTION],
}
```

On the other hand, the `CODING_ASSISTANT` use case is designed for instances
where there is a needs to protect against malicious code attacks. This is
conveniently configured by setting a ScannerType.CODE_SHIELD for both the
`ASSISTANT` and `TOOL` roles:

```python
UseCase.CODING_ASSISTANT: {
    Role.ASSISTANT: [ScannerType.CODE_SHIELD],
    Role.TOOL: [ScannerType.CODE_SHIELD],
}
```

### Initialize LlamaFirewall with a Use Case

To initialize LlamaFirewall with a specific use case, you can use the
`from_usecase` method. This method takes in a `UseCase` enum value as an
argument and returns an instance of LlamaFirewall configured with the
corresponding scanning layers and scanners. For example:

```python
from security.llamafirewall_pre_oss.llamafirewall import LlamaFirewall, UseCase

# Initialize LlamaFirewall with the code assistant use case
llamafirewall = LlamaFirewall.from_usecase(UseCase.CODING_ASSISTANT)

# Plug in an assistant and start scanning its input with LlamaFirewall
# ...agent code...
async for output in agent.stream:
    # Scan the output with LlamaFirewall
    lf_result: ScanResult = await llamafirewall.scan_async(
    Message(
        role=Role.ASSISTANT,
        content= output,
    )
    )

    logging.info(f"Scan result: {lf_result}")
    # continue processing the output...
    yield output
```

### Defining an Additional Use Case

To create an instance of LlamaFirewall with a custom use case, you can define an additional use case by adding it to the `PREDEFINED_USE_CASES` dictionary. This allows you to specify the scanning layers and scanners that should be used for your particular use case.

For example, let's say you want to create a use case for a web application that requires additional scanning for a set of specific scanners you've customized. You could define this use case as follows:

```python
from enum import Enum

class UseCase(Enum):
    # ... existing use cases ...
    WEB_APPLICATION = "web_application"

# Define the scanning layers and scanners for the new use case
PREDEFINED_USE_CASES[UseCase.WEB_APPLICATION] = {
    Role.USER: [ScannerType.MY_MALICIOUS_UPLOADS_SCANNER],
    Role.SYSTEM: [ScannerType.MY_MISALIGNMENT_SCANNER],
    # ... additional scanning layers and scanners for your web application use case ...
}
```

With this additional use case defined, you can now create an instance of LlamaFirewall using the `from_usecase` method:

```python
firewall = LlamaFirewall.from_usecase(UseCase.WEB_APPLICATION)
```

This will easily provide you with a LlamaFirewall instance configured with the scanning layers and scanners you've specified for your web application use case.
