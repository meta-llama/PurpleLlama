/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <string.h>

void WarnOnMisuseofSnprintf() {
  char buf[50];
  char str[60];
  int ret = 0;
  // ruleid: bugprone-snprintf
  ret = snprintf(buf, sizeof(buf), "%s", str);
  buf[ret] = '\0';

  // ruleid: bugprone-snprintf
  ret = snprintf(buf, sizeof(buf), "%s", str);
  *(buf + ret) = '\0';
}

void NoWarnOnNoMisuseInBinaryOperator() {
  char buf[50];
  char str[60];
  int ret = 0;
  ret = snprintf(buf, sizeof(buf), "%s", str);
  // ok: bugprone-snprintf
  if (ret < sizeof(buf)) {
    printf("ret is greater");
  } else {
    printf("%d", ret);
  }
}

void NoWarnOnNonMisuseInDeclarationStatement() {
  char divider[101];
  char color[10] = "adfadf";
  int bytes_read = snprintf(divider, sizeof(divider), "%s", color);
  // ok: bugprone-snprintf
  if (bytes_read < sizeof(divider)) {
    memset(divider + bytes_read, '-', sizeof(divider) - bytes_read);
  }
  divider[100] = '\0';
}

void NoWarnOnNonStringFormatType(int a) {
  char buf[50];
  char str[60];
  int ret = 0;
  ret = snprintf(buf, sizeof(buf), "%d", a);
  // ok: bugprone-snprintf
  printf("%d", ret);
}

void NoWarnOnMisuseErrorValidationInBinaryOperator() {
  char buf[50];
  char str[60];
  int ret = 0;
  ret = snprintf(buf, sizeof(buf), "%s", str);
  // ok: bugprone-snprintf
  if (os_snprintf_error(sizeof(buf), ret)) {
    printf("ret is greater");
  } else {
    printf("%d", ret);
  }
}

void NoWarnOnMisuseInUsageWithIncorrectErrorValidationInBinaryOperator() {
  char buf[50];
  char str[60];
  int ret = 0;
  int x = 0;
  ret = snprintf(buf, sizeof(buf), "%s", str);
  // ok: bugprone-snprintf
  if (true) {
    if (10 < ret) {
      printf("ret is greater");
    }
  }
}
