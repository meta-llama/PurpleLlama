#include "llamafirewall_types.h"
#include <string.h> // For strcmp
#include <stdio.h>  // For NULL if not in string.h for some compilers

// --- ScannerType ---
CScannerType c_scanner_type_from_string(const char* s) {
    if (!s) return SCANNER_TYPE_UNKNOWN;
    if (strcmp(s, "code_shield") == 0) return SCANNER_TYPE_CODE_SHIELD;
    if (strcmp(s, "prompt_guard") == 0) return SCANNER_TYPE_PROMPT_GUARD;
    if (strcmp(s, "agent_alignment") == 0) return SCANNER_TYPE_AGENT_ALIGNMENT;
    if (strcmp(s, "hidden_ascii") == 0) return SCANNER_TYPE_HIDDEN_ASCII;
    if (strcmp(s, "pii_detection") == 0) return SCANNER_TYPE_PII_DETECTION;
    if (strcmp(s, "regex") == 0) return SCANNER_TYPE_REGEX;
    return SCANNER_TYPE_UNKNOWN;
}

const char* c_scanner_type_to_string(CScannerType type) {
    switch (type) {
        case SCANNER_TYPE_CODE_SHIELD: return "code_shield";
        case SCANNER_TYPE_PROMPT_GUARD: return "prompt_guard";
        case SCANNER_TYPE_AGENT_ALIGNMENT: return "agent_alignment";
        case SCANNER_TYPE_HIDDEN_ASCII: return "hidden_ascii";
        case SCANNER_TYPE_PII_DETECTION: return "pii_detection";
        case SCANNER_TYPE_REGEX: return "regex";
        default: return "unknown_scanner_type";
    }
}

// --- ScanDecision ---
CScanDecision c_scan_decision_from_string(const char* s) {
    if (!s) return SCAN_DECISION_ERROR;
    if (strcmp(s, "allow") == 0) return SCAN_DECISION_ALLOW;
    if (strcmp(s, "human_in_the_loop_required") == 0) return SCAN_DECISION_HUMAN_IN_THE_LOOP_REQUIRED;
    if (strcmp(s, "block") == 0) return SCAN_DECISION_BLOCK;
    return SCAN_DECISION_ERROR; // Default to error if unknown string
}

const char* c_scan_decision_to_string(CScanDecision decision) {
    switch (decision) {
        case SCAN_DECISION_ALLOW: return "allow";
        case SCAN_DECISION_HUMAN_IN_THE_LOOP_REQUIRED: return "human_in_the_loop_required";
        case SCAN_DECISION_BLOCK: return "block";
        case SCAN_DECISION_ERROR: return "error"; // Explicitly handle error case
        default: return "unknown_decision";
    }
}

// --- CRole ---
CRole c_role_from_string(const char* s) {
    if (!s) return ROLE_UNKNOWN;
    if (strcmp(s, "tool") == 0) return ROLE_TOOL;
    if (strcmp(s, "user") == 0) return ROLE_USER;
    if (strcmp(s, "assistant") == 0) return ROLE_ASSISTANT;
    if (strcmp(s, "memory") == 0) return ROLE_MEMORY;
    if (strcmp(s, "system") == 0) return ROLE_SYSTEM;
    return ROLE_UNKNOWN;
}

const char* c_role_to_string(CRole role) {
    switch (role) {
        case ROLE_TOOL: return "tool";
        case ROLE_USER: return "user";
        case ROLE_ASSISTANT: return "assistant";
        case ROLE_MEMORY: return "memory";
        case ROLE_SYSTEM: return "system";
        default: return "unknown_role";
    }
}

// --- CScanStatus ---
CScanStatus c_scan_status_from_string(const char* s) {
    if (!s) return SCAN_STATUS_ERROR;
    if (strcmp(s, "success") == 0) return SCAN_STATUS_SUCCESS;
    if (strcmp(s, "error") == 0) return SCAN_STATUS_ERROR;
    if (strcmp(s, "skipped") == 0) return SCAN_STATUS_SKIPPED;
    return SCAN_STATUS_ERROR; // Default to error
}

const char* c_scan_status_to_string(CScanStatus status) {
    switch (status) {
        case SCAN_STATUS_SUCCESS: return "success";
        case SCAN_STATUS_ERROR: return "error";
        case SCAN_STATUS_SKIPPED: return "skipped";
        default: return "unknown_status";
    }
}
