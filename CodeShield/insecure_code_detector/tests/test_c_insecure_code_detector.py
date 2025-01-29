# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
C_REGEX_TEST_CASES = [
    ("strcpy(dest, src)", 2),
    ("// strcpy(dest, src)", 0),
    ('scanf("%s", buffer)', 1),
    ("strcat(dest, source)", 1),
    ("getpw(buff, buffer)", 1),
    ("hello", 0),
    ("MD5_Init(", 0),
    ("BF_encrypt()", 1),
    ("BF_decrypt()", 1),
    ("EVP_bf_ecb(", 2),
    ("EVP_aes_256_gcm()", 0),
    ("EVP_des_cbc()", 2),
    ("BN_get0_nist_prime_192()", 1),
    ("gets(str);", 1),
    ("temp_filename = tmpnam(NULL);, ", 1),
    ("char *token = strtok(str, delim);", 1),
]

C_SEMGREP_TEST_CASES = [
    # Sprintf and Snprintf
    ("ret = snprintf(dest, 10, src); dest[ret] = '1';", 1),
    ("ret = snprintf(dest, 10, src);", 0),
    ("ret = snprintf(dest, 10, src); if (ret < 10) { dest[ret] = '1'; }", 0),
    ("ret = snprintf(dest, 10, src); if (ret < 10) { *(dest + ret); }", 0),
    ("ret = snprintf(dest, 10, src); if (src < 10) { *(dest + ret); }", 0),
    ("ret = snprintf(dest, 10, src); if (dest < 10) { *(dest + ret); }", 0),
    ('sprintf(dest, "%s", buffer)', 1),
    # Integer overflow
    ("ptr = malloc(n * 5);", 1),
    ("if (n < SIZE_MAX/5) { ptr = malloc(n * 5); }", 0),
    ("ptr = malloc(WIDTH * 5)", 0),
    # Invalid free
    ("func foo() { char buf[10]; free(buf);}", 1),
    ("func foo() { char ptr; free(&ptr);}", 1),
    # Stack buffer overflow
    ("char buf[100]; int n = func(); memcpy(buf, src, n);", 1),
    ("char buf[100]; int n = func(); memmove(buf, src, n);", 1),
    ("char buf[100]; int n = func(); if (n < 100) { memcpy(buf, src, n); }", 0),
    ("void func(int n) { char buf[100]; memcpy(buf, src, n); }", 1),
    ("char buf[100]; strcpy(buf, src);", 2),
    # Fixed key/secret/seeds/ randomness
    ("return rand()", 1),
    ("srand()", 0),
    ('const char* secret = "abc";', 1),
    ('const char* private_key = "abc";', 1),
    ('const char* nonce = "abc";', 1),
    ('const char* prng_seed = "abc";', 1),
    ('const char* regularVariable = "abc";', 0),
    ("const int secret = 15;", 1),
    ("int anotherVar = 1; const int iv = anotherVar;", 1),
    # RSA Key generation
    ("RSA_generate_key_ex(ctx, 1024, exponent, nullptr);", 1),
    ("RSA_generate_key(ctx, 768, exponent, nullptr);", 1),
    ("RSA_generate_key_ex(ctx, 2048, exponent, nullptr);", 0),
    # Other
    ('system("ls -l");', 0),
    ("system(F_CLEAR);", 0),
    ("system(user_input);", 1),
]


class TestCInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_c_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.C, C_REGEX_TEST_CASES)

    async def test_c_semgrep_patterns(self) -> None:
        await self.run_semgrep_tests(Language.C, C_SEMGREP_TEST_CASES)
