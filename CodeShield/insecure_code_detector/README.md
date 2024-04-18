# Insecure Code Detector

Insecure Code Detector(ICD) is a library developed to identify Insecure coding practices in any given code snippet. ICD operates using a set of analyzers, each equipped with specific ruleset. As of now, ICD supports 8 different programming languages. Theses include widely-used languages such as C, C++, Python, and Java.

## Building principles of ICD
ICD is built on three fundamental principles-

- Robustness: ICD was mainly built to analyse LLM generated code. Since LLM generated code might be incomplete and non-parseable, robustness was a critical consideration during ICD's development.

- Speed: We want ICD to be considerably fast unlike complex tools with taint analysis which takes several hours to finish analysis.

- Extensibility: ICD is designed with extensibility in mind. It is built to accommodate the easy addition of more analyzers and language support.

## Non goals of ICD

- Vulnerability Identifier: ICD is not designed to serve as a comprehensive static analysis tool for identifying vulnerabilities. Its primary focus is on detecting insecure coding practices rather than pinpointing specific vulnerabilities.Insecure coding practices refer to any coding style or practice that requires detailed attention from the developer to ensure it is written securely. These practices may not necessarily lead to immediate vulnerabilities, but they could potentially create security risks if not addressed properly.

## How does it work?

ICD comprises of a set of analyzers which independently assess the security of the input code snippet based on static analysis rules. Currently we support [regex](https://en.wikipedia.org/wiki/Regular_expression) and [semgrep](https://github.com/semgrep/semgrep) analyzers. We plan to add support for more analyzers.

## Languages Supported

- `C`
- `C++`
- `C#`
- `Java`
- `Javascript`
- `Python`
- `PHP`
- `Rust`

## Limitations

- Since we have optimized for speed and robustnesss, ICD unfortunately doesn't work well for vulnerability categories which require taint flow analysis for high accuracy.
- ICD operates on a "best guess" basis, aiming to identify insecure callsites even within incomplete code. This approach, while effective in many cases, can lead to false positives (FPs). These are instances where the tool flags a piece of code as potentially insecure, even though it may not actually pose a security risk.

## How can I help improve?

Feel free to contribute to improve ICD by
  - adding new analyzers suitable for the requirements
  - adding new rules or iterating on existing rules to improve quality
  - adding more language support

Please ensure to add test cases while iterating on the rules.
