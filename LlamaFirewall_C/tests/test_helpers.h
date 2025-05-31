#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../src/llamafirewall_types.h" // For CMessage, CRole, CScanResult etc.

// Define colors for test output (optional, works on ANSI terminals)
#define KGRN  "\x1B[32m"
#define KRED  "\x1B[31m"
#define KNRM  "\x1B[0m"

#define TEST_PASSED(test_name) printf("%s[PASS]%s %s\n", KGRN, KNRM, test_name)
#define TEST_FAILED(test_name, reason) printf("%s[FAIL]%s %s - Reason: %s\n", KRED, KNRM, test_name, reason); test_suite_failed = 1

static int test_suite_failed = 0;

static inline CMessage create_test_message(CRole role, const char* content) {
    CMessage msg;
    msg.role = role;
    msg.content = strdup(content);
    return msg;
}

static inline void free_test_message_content(CMessage* msg) {
    if (msg && msg->content) {
        free(msg->content);
        msg->content = NULL;
    }
}

// Local version for tests, as original is static in llamafirewall_core.c
static inline void free_scan_result_members(CScanResult* result) {
    if (result && result->reason) {
        free(result->reason);
        result->reason = NULL;
    }
}
