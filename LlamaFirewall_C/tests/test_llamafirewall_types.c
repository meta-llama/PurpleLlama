#include "../src/llamafirewall_types.h"
#include "test_helpers.h"

void test_role_conversions() {
    const char* test_name = "RoleConversions";
    if (c_role_from_string("user") == ROLE_USER && strcmp(c_role_to_string(ROLE_USER), "user") == 0 &&
        c_role_from_string("assistant") == ROLE_ASSISTANT && strcmp(c_role_to_string(ROLE_ASSISTANT), "assistant") == 0 &&
        c_role_from_string("invalid_role") == ROLE_UNKNOWN && strcmp(c_role_to_string(ROLE_UNKNOWN), "unknown_role") == 0) {
        TEST_PASSED(test_name);
    } else {
        TEST_FAILED(test_name, "Mismatch in role string/enum conversion");
    }
}

void test_decision_conversions() {
    const char* test_name = "DecisionConversions";
     if (c_scan_decision_from_string("allow") == SCAN_DECISION_ALLOW && strcmp(c_scan_decision_to_string(SCAN_DECISION_ALLOW), "allow") == 0 &&
        c_scan_decision_from_string("block") == SCAN_DECISION_BLOCK && strcmp(c_scan_decision_to_string(SCAN_DECISION_BLOCK), "block") == 0 &&
        c_scan_decision_from_string("human_in_the_loop_required") == SCAN_DECISION_HUMAN_IN_THE_LOOP_REQUIRED &&
         strcmp(c_scan_decision_to_string(SCAN_DECISION_HUMAN_IN_THE_LOOP_REQUIRED), "human_in_the_loop_required") == 0 &&
        c_scan_decision_from_string("invalid_decision") == SCAN_DECISION_ERROR && strcmp(c_scan_decision_to_string(SCAN_DECISION_ERROR), "error") == 0 ) {
        TEST_PASSED(test_name);
    } else {
        TEST_FAILED(test_name, "Mismatch in decision string/enum conversion");
    }
}

int main() {
    printf("--- Running LlamaFirewall Types Tests ---\n");
    test_suite_failed = 0; // Reset for this test suite
    test_role_conversions();
    test_decision_conversions();
    if (test_suite_failed) {
        printf("%s[RESULT] Some LlamaFirewall Types tests FAILED.%s\n", KRED, KNRM);
        return 1;
    } else {
        printf("%s[RESULT] All LlamaFirewall Types tests PASSED.%s\n", KGRN, KNRM);
        return 0;
    }
}
