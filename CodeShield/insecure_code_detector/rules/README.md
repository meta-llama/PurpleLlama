# Security rules for Insecure Code Detector

This directory is equipped with a set of rules that serve as security signals, designed to identify insecure coding practices. These rules essentially fuel the 'Insecure Code Detector'. To improve patterns, add/edit entries in the yaml file of the corresponding language. If you want to add a new language, create a new yaml file with language as the file name.

## Adding Regex Rules

Given below is the structure of each pattern. Add the regex pattern in `rule` field and other fields capture the metadata of the pattern.

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
    description: Potential buffer overflow due to use of strcpy
    rule: \bstrcpy\s*\(
    severity: Error
    pattern_id: vulnerable-strcpy
```

Severity:
Convention for severity and all the severity levels are as follows -
`Error` > `Warning` > `Advice` > `Disabled`

### Testing Regex Rules

Add unit test cases under language specific file of test directory and invoke test cases

## Adding Semgrep Rule

Add semgrep rule along with the corresponding test case file in the `semgrep/` directory under specific language. Follow https://semgrep.dev/docs/writing-rules/overview/ for guidance on writing rules.

### Testing Semgrep Rule

Run semgrep against unit test files added. Additionally you can add unit tests in ICD and test out the patterns.
