/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include "stdio.h"
#include "stdlib.h"

struct SkippedObject {
  int e;
};

struct OtherObject {
  int c;
  int d;
};

struct SomeObject {
  int a;
  int b;
  float c;
  OtherObject* other;
  SkippedObject* skipped;
  char tag[4];
  const char* name;
};

enum Foo {
  FOO = 1,
  BAR = 2,
};

void some_fn(SomeObject* x) {
  printf("x->a: %d\n", x->a);
  printf("x->b: %d\n", x->b);
  printf("x->c: %.5f\n", x->c);
  printf("x->other->c: %d\n", x->other->c);
  printf("x->other->d: %d\n", x->other->d);
  printf("x->tag: %.*s\n", 4, &x->tag[0]);
  printf("x->name: %s\n", x->name);
}

void some_template_fn(const std::vector<OtherObject>& vec) {
  for (auto& x : vec) {
    printf("x.c: %d\n", x.c);
    printf("x.d: %d\n", x.d);
  }
}

void some_more_classes(const std::map<int, std::string>& map) {
  for (auto& pair : map) {
    printf("k: %d, v: %s\n", pair.first, pair.second.c_str());
  }
}

void enum_and_ptrs(
    Foo foo,
    const OtherObject* other_1,
    const OtherObject* other_2) {
  printf("foo: %d\n", foo);
  printf("other_1->c: %d\n", other_1->c);
  printf("other_1->d: %d\n", other_1->d);
  printf("other_2->c: %d\n", other_2->c);
  printf("other_2->d: %d\n", other_2->d);
}

using SomeTypeDefEnum = Foo;

void test_typedef_enum(SomeTypeDefEnum foo) {
  printf("foo: %d\n", foo);
}
