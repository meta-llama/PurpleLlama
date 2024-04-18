/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

int func(int n) {
  char buf[100];
  // ruleid: write-to-stack-buffer
  memcpy(buf, src, n);
  return n;
}

void func2() {
  char buf[100];
  int n = func();
  if (n < 100) {
    // ok: write-to-stack-buffer
    memcpy(buf, src, n);
  }
}

int func3(int n) {
  char buf[100];
  // ok: write-to-stack-buffer
  memcpy(buf, "hello", n);
  return n;
}

int func4(char* buf, char* src, int n) {
  // ok: write-to-stack-buffer
  memcpy(buf, src, n);
  return n;
}
