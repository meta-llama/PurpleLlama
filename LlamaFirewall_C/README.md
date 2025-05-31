# LlamaFirewall C Implementation

This project is a C-based implementation of the LlamaFirewall, designed to be run as a D-Bus service, primarily for Yocto Linux environments.

## Overview

The LlamaFirewall C service provides content scanning capabilities similar to the original Python LlamaFirewall. It exposes its functionality via a D-Bus interface, allowing other applications on the system to request scans of messages.

Key components:
-   **Core Logic (`llamafirewall_core.c`):** Handles message processing, scanner orchestration, and decision aggregation.
-   **Scanners:**
    -   **RegexScanner (`scanners/regex_scanner.c`):** Performs regex-based pattern matching. Uses the PCRE2 library.
    -   **PromptGuardScanner (`scanners/prompt_guard_scanner.c`):** Intended for detecting prompt injections and harmful content using ML models. In this C version, it currently **simulates** an Inter-Process Communication (IPC) call to a Python helper script (`promptguard_helper.py`) for the actual model inference. This hybrid approach is used due to the complexity of porting PyTorch/Transformer models directly to C.
    -   Other scanners can be added following these patterns.
-   **D-Bus Service (`llamafirewall_dbus_service.c`):** Exposes the firewall via D-Bus (`io.LlamaFirewall` service name) using the `sd-bus` library. The primary method is `ScanMessage`.
-   **Data Types (`llamafirewall_types.c`):** Defines C equivalents of the Python project's data structures.

## Building

The project uses a `Makefile` located in the `src/` directory.

### Prerequisites
-   GCC (or compatible C compiler)
-   Make
-   `pkg-config`
-   `pcre2` library (libpcre2-dev or equivalent)
-   `systemd` library (libsystemd-dev or equivalent, for sd-bus)

### Compilation
From the `LlamaFirewall_C/src/` directory:
```bash
make
```
This will produce the `llamafirewall-dbus-service` executable.

To build the unit tests:
```bash
make test_TARGET # e.g., make test_regex_scanner, make test_llamafirewall_types
```
To run all C unit tests:
```bash
make test
```

## Yocto Integration

A Yocto recipe is provided in `recipes-security/llamafirewall-c/llamafirewall-c_0.1.bb`. This recipe handles:
-   Building the C executable.
-   Installing the executable to `/usr/bin/`.
-   Installing the D-Bus service file (`conf/io.LlamaFirewall.service`) to `/usr/share/dbus-1/system-services/`.
-   Installing the dummy `promptguard_helper.py` to `/usr/lib/llamafirewall-c/` (path may vary based on Yocto config for `nonarch_base_libdir`).
-   Managing dependencies (`pcre2`, `systemd`, `python3`).

## D-Bus API

-   **Service:** `io.LlamaFirewall`
-   **Object Path:** `/io/LlamaFirewall`
-   **Interface:** `io.LlamaFirewall.Service1`

### Method: `ScanMessage`
-   **Input Signature:** `ssas` (string `message_content`, string `message_role`, array of structs `trace_history`)
    -   `message_content`: The text content of the message to scan.
    -   `message_role`: The role of the message (e.g., "user", "assistant", "tool", "system", "memory").
    -   `trace_history`: An array of past messages. Each struct is `(ss)`: (string `role`, string `content`). Pass an empty array if no trace.
-   **Output Signature:** `ssds` (string `decision`, string `reason`, double `score`, string `status`)
    -   `decision`: "allow", "block", "human_in_the_loop_required", or "error".
    -   `reason`: Explanation for the decision.
    -   `score`: Confidence score (0.0 to 1.0).
    -   `status`: "success" or "error".

### Example `busctl` usage:
```bash
busctl call io.LlamaFirewall /io/LlamaFirewall io.LlamaFirewall.Service1 ScanMessage ssas   "Hello, world!"   "user"   0
```

## Future Development / Refinements

-   **Real IPC for PromptGuardScanner:** Implement the actual IPC mechanism (e.g., using `popen` or sockets) to call the Python `promptguard_helper.py` script. The Python script itself needs to be updated to perform real model inference.
-   **Implement Other Scanners:** Port or create hybrid wrappers for other scanners from the original LlamaFirewall (CodeShield, PIIDetection, etc.).
-   **Configuration Management:** Currently, scanner configurations (like which scanners run per role, or PromptGuard thresholds) are largely hardcoded in C. This could be moved to a configuration file.
-   **Advanced Error Handling:** More detailed error reporting over D-Bus.
-   **Logging:** Integrate more structured logging (e.g., syslog).
-   **Comprehensive Unit Tests:** Expand C unit test coverage, especially for `llamafirewall_core.c`.
-   **Systemd Service Unit:** Provide a systemd `.service` file for managing the `llamafirewall-dbus-service` daemon lifecycle if more control is needed than D-Bus activation provides.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
