/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

typedef const void* MCFStringRef;
typedef unsigned int MCFStringEncoding;

bool MCFStringGetCString(
    MCFStringRef str,
    char* buffer,
    int bufferSize,
    MCFStringEncoding encoding);

void WarnOnIgnoredResultSuggestFix(MCFStringRef strref) {
  char buffer[256];
  // ruleid: mcf-get-cstring-result-ignored
  MCFStringGetCString(strref, buffer, 256, 0x08000100);
}

char* WarnOnIgnoredResultSuggestFix2(MCFStringRef strref) {
  char buffer[256];
  // ruleid: mcf-get-cstring-result-ignored
  MCFStringGetCString(strref, buffer, 256, 0x08000100);
}

bool WarnOnIgnoredResultSuggestFix3(MCFStringRef strref) {
  char buffer[256];
  // ruleid: mcf-get-cstring-result-ignored
  MCFStringGetCString(strref, buffer, 256, 0x08000100);
}

double WarnOnIgnoredResultSuggestFix4(MCFStringRef strref) {
  char buffer[256];
  // ruleid: mcf-get-cstring-result-ignored
  MCFStringGetCString(strref, buffer, 256, 0x08000100);
}

void NoWarnOnAssignedResult(MCFStringRef strref) {
  char buffer[256];
  // ok: mcf-get-cstring-result-ignored
  bool ret = MCFStringGetCString(strref, buffer, 256, 0x08000100);
}

void NoWarnOnIfCheck(MCFStringRef strref) {
  char buffer[256];
  // ok: mcf-get-cstring-result-ignored
  if (MCFStringGetCString(strref, buffer, 256, 0x08000100)) {
    return;
  }
  // ok: mcf-get-cstring-result-ignored
  if (!MCFStringGetCString(strref, buffer, 256, 0x08000100)) {
    return;
  }
}

bool NoWarnOnReturn(MCFStringRef strref) {
  char buffer[256];
  return MCFStringGetCString(strref, buffer, 256, 0x08000100);
}

int check(bool);
bool NoWarnOnOtherFunctionCall(MCFStringRef strref) {
  char buffer[256];
  check(MCFStringGetCString(strref, buffer, 256, 0x08000100));
}
