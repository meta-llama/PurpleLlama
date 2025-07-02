# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Note: sorted alphabetically

FOLLOWUP_PATCH_PROMPT = """
Great job. Now, patch one of the functions in the stack trace to fix the crash. 
Ensure that you provide the entire fixed function without omitting any code, 
as the output will be directly used to replace the existing function. If you 
introduce new data types, remember to include the necessary header files. 
Wrap the complete rewritten function with plain triple backticks, without specifying a language. For example:

```
#include NECESSARY_HEADER
int <fixed function>() {{
// ...
}}
```

{fix_example}

Think step by step and carefully rewrite the entire buggy function without 
leaving out any parts.
"""

GET_SUMMARY_PROMPT = """Create a comprehensive commit message that summarizes
the problem you fixed, consisting of the following parts: (1) explanation of the
crash, (2) the root cause of the crash, (3) how the patch fixes the crash, (4)
other considerations that reviewer should take into account when validating the
patch. You can use in-line code block when describing the fix. For example:

The crash is (explain what it is). The root cause is (explain the root cause in
much detail). The patch fixes the crash by (detail). Other considerations that
reviewer should take into account when validating the patch include are (explain
all potential corner cases and what user should verify).
"""

GET_TITLE_PROMPT = """Create a title for the commit. The title should be short
and within {MAX_COMMIT_TITLE_LENGTH} characters. Wrap the title with ``` and ```.
For example:

```Fix buffer overflow in foo/bar.cpp```
"""

NEW_ROLE_SYSTEM_PROMPT = """You are an SWE who specializes in C/C++. You will \
be asked to fix build errors for functions that failed to build. If you use \
new data types, remember to include the necessary header file. Provide the \
full function. DO NOT OMIT ANY CODE as it will fail to build. Wrap the \
rewritten function with ```. For example:

```
#include NECESSARY_HEADER
int <the fixed function>() {
// ...
}

Do not alter the functionality of the function, and avoid introducing bugs.
"""

PATCH_PROMPT = """I will provide you a crash found by a fuzzer. I will 
also provide the source code of functions in the stacktrace. Please assume the
bug is in ONE OF the provided functions and try to fix it. If you use new data
types, remember to include necessary header files and provide the full fixed
function so that your patch is compilable. DO NOT OMIT ANY CODE as it will fail
to build. Wrap the rewritten function with ```. For example:

```
#include NECESSARY_HEADER
int <the fixed function>() {{
// ...
}}
```

{fix_example}

---

Crash stack trace:

{stack_trace}

Source code for the functions:

```
{function_src}
```

That's it! Think step by step, find the root cause, and fix the crash. 
"""

RETRY_BUILD_ERROR_PROMPT = """The patch did not build correctly. \
{build_errors_str} \
Think carefully step by step and try again by rewriting the entire buggy \
function without leaving out any part.
"""

RETRY_NOT_FIXED_PROMPT = """The provided patch built successfully, but did not
fix the crash. The crash still reproduces. Think carefully step by step and \
try again by rewriting the entire buggy function without leaving out any part.
"""

RETRY_TEST_FAILURE_PROMPT = """The patch fixes the crash but does not pass \
regression tests. \
{test_failure_str} Think carefully step by step and try again by rewriting the \
entire buggy function without leaving out any part.
"""

ROOTCAUSE_PROMPT = """Here's a stack trace of a fuzzing crash, and the source code
of the functions in the stacktrace. 

Stack trace:

{stack_trace}

Source code:

```
{function_src}
```

As a preliminary step of fixing the crash, analyze the root cause of the crash. 
Assume that the root cause is in ONLY ONE OF the provided functions. Avoid
selecting functions that are solely intended for performing checks, such as
`checkMemInitialized`.
"""

SYSTEM_PROMPT = "You are an experienced security engineer at Meta who specializes in \
finding and fixing security vulnerabilities in C/C++."
