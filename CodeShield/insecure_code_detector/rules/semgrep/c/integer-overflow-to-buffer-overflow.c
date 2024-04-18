/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>

void test1(int x) {
  // ruleid: potential-integer-overflow-to-buffer-overflow
  char* buf = (char*)malloc(x * 2); // Integer overflow
  if (buf != NULL) {
    // Buffer overflow
    buf[x] = 'a';
  }
}

void test2(int x) {
  if (x < 100) {
    // ok: potential-integer-overflow-to-buffer-overflow
    char* buf = (char*)malloc(x * 2);
    if (buf != NULL) {
      buf[x] = 'a';
    }
  }
}

void test3(int x) {
  if (x > 100) {
    return;
  }
  // ok: potential-integer-overflow-to-buffer-overflow
  char* buf = (char*)malloc(x * 2);
  if (buf != NULL) {
    buf[x] = 'a';
  }
}

void test4(int x) {
  int CONSTANT = 100;
  // ok: potential-integer-overflow-to-buffer-overflow
  char* buf = (char*)malloc(CONSTANT * 2);
  if (buf != NULL) {
    buf[x] = 'a';
  }
}
