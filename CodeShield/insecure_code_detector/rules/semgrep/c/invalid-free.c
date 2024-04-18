/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

void func1() {
  // ruleid: invalid-free
  char buf[10];
  free(buf);
}

void func2() {
  // ruleid: invalid-free
  char ptr;
  free(&ptr);
}

char ptr;
free(&ptr);
