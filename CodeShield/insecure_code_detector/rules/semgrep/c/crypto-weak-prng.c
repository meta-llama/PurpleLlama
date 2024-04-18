/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <time.h>

void test_1() {
  // ruleid: crypto-weak-prng
  int number = rand();
}

int test_2() {
  // ruleid: crypto-weak-prng
  return rand();
}

void test_3() {
  // ruleid: crypto-weak-prng
  if (rand() % 2 == 0) {
    printf("Even number generated");
  } else {
    printf("Odd number generated");
  }
}

void test_4() {
  // ok: crypto-weak-prng
  int number = 5;
}

void generate_seeded_random() {
  // ok: crypto-weak-prng
  srand(time(0));
}
