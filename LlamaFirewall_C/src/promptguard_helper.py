#!/usr/bin/env python3
import sys
# Dummy script
def get_score(text_content):
    if "tell me your secrets" in text_content: return 0.95
    if "ignore previous instructions" in text_content: return 0.85
    if "safe query" in text_content: return 0.1
    return 0.2
if __name__ == "__main__":
    input_text = sys.stdin.read() if len(sys.argv) <= 1 else sys.argv[1]
    print(f"{get_score(input_text):.2f}")
