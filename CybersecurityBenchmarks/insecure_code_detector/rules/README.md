# Security rules for Insecure Code Detector

This directory is equipped with a set of rules that serve as security signals, designed to identify insecure coding practices. These rules essentially fuel the 'Insecure Code Detector'. To improve patterns, add/edit entries in the yaml file of the corresponding language. If you want to add a new language, create a new yaml file with language as the file name.

Structure of each pattern:
```
    description: "",
    cwe_id: "",
    rule: "",
    severity: "",
    pattern_id: "",
 ```
Example:
```
    cwe_id: CWE-120
    description: Potential buffer overflow due to use of sprintf
    rule: '{ sprintf(_);}'
    severity: Warning
    pattern_id: C-W-001
```

Severity:
Convention for severity and all the severity levels are as follows -
`Error` > `Warning` > `Advice` > `Disabled`
