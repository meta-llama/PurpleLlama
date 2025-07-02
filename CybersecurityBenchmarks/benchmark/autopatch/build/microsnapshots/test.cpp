/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test.h"

int main(int argc, char** argv) {
  SomeObject* x = (SomeObject*)calloc(1, sizeof(SomeObject));
  OtherObject* y = (OtherObject*)calloc(1, sizeof(OtherObject));
  SkippedObject* z = (SkippedObject*)calloc(1, sizeof(SkippedObject));
  z->e = 1;
  x->other = y;
  x->skipped = z;
  x->a = 1337;
  x->b = 42;
  x->c = 1.234567;
  x->name = "foo";
  memcpy(&x->tag[0], "TAGX", 4);
  y->c = 1;
  y->d = -1;
  some_fn(x);
  some_template_fn({{.c = 3, .d = 4}});
  some_more_classes({{42, "some string"}});
  enum_and_ptrs(Foo::BAR, y, y);
  test_typedef_enum(Foo::FOO);
  return 0;
}
