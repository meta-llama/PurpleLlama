/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>

class SomeObject {
 public:
  int some_noise;
  int some_value;
};

int gettime() {
  struct timespec tt = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &tt);
  return tt.tv_nsec;
}

#ifdef _PATCHED_DELTA
int someFunction(std::string input, SomeObject* obj, int* local) {
  *local = gettime();
  obj->some_noise = gettime();
  obj->some_value = input.length();
  if (input == "foo") {
    obj->some_value = input.length() + 1;
  }
  return 3;
}
#else
int someFunction(std::string input, SomeObject* obj, int* local) {
  *local = gettime();
  obj->some_noise = gettime();
  obj->some_value = input.length();
  return 3;
}
#endif

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
  if (Size == 0) {
    return 0;
  }
  int noisy_local = 0;
  int n = 0;
  std::string k((char*)Data, Size);
  SomeObject obj = {.some_noise = 0, .some_value = 0};
  n = someFunction(k, &obj, &noisy_local);
  printf(
      "obj.some_value: %d for %s returns %d\n", obj.some_value, k.c_str(), n);
  return 0;
}
