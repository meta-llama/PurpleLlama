#include "llamafirewall_core.h"
#include "llamafirewall_types.h" // Already includes stddef.h for size_t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "scanners/regex_scanner.h" // For regex_scanner_scan, init, cleanup
#include "scanners/prompt_guard_scanner.h" // For prompt_guard_scanner_scan, init, cleanup

// --- End Placeholder Scanner Implementations ---


// Helper to free CScanResult members
void free_scan_result_members(CScanResult* result) {
    if (result) {
        free(result->reason); // free strdup'd string
        result->reason = NULL;
    }
}

int llamafirewall_core_init(LlamaFirewallCore* core) {
    if (!core) return -1;
    core->initialized = 0; // Set to 0 initially

    // Initialize Regex Scanner
    if (regex_scanner_init(&core->regex_scanner_config) != 0) {
        fprintf(stderr, "Failed to initialize RegexScanner.\n");
        return -1;
    }

    // Initialize PromptGuard Scanner
    // Define a path for the dummy Python script and a block threshold.
    // These could come from a config file or environment variables in a real app.
    const char* prompt_guard_script_path = "/opt/llamafirewall/dummy_prompt_guard.py"; // Example path
    double prompt_guard_threshold = 0.8; // Example threshold
    if (prompt_guard_scanner_init(&core->prompt_guard_config, prompt_guard_script_path, prompt_guard_threshold) != 0) {
        fprintf(stderr, "Failed to initialize PromptGuardScanner.\n");
        regex_scanner_cleanup(&core->regex_scanner_config); // Clean up already initialized scanners
        return -1;
    }

    core->initialized = 1; // Set to 1 on successful initialization of all components
    printf("LlamaFirewallCore initialized successfully.\n");
    return 0;
}

void llamafirewall_core_cleanup(LlamaFirewallCore* core) {
    if (!core) return;
    regex_scanner_cleanup(&core->regex_scanner_config);
    prompt_guard_scanner_cleanup(&core->prompt_guard_config);
    core->initialized = 0;
    printf("LlamaFirewallCore cleaned up.\n");
}

CScanResult llamafirewall_scan_message(
    LlamaFirewallCore* core,
    const CMessage* input_message,
    const CTrace* input_trace
) {
    if (!core || !core->initialized || !input_message) {
        return (CScanResult){
            .decision = SCAN_DECISION_ERROR,
            .reason = strdup("Core not initialized or invalid input"),
            .score = 0.0,
            .status = SCAN_STATUS_ERROR
        };
    }

    printf("Scanning message for role: %s\n", c_role_to_string(input_message->role));

    // Define which scanners to run for each role (simplified)
    GenericScannerFunc active_scanners[5];
    void* active_configs[5];
    int num_scanners = 0;

    switch (input_message->role) {
        case ROLE_USER:
            active_scanners[num_scanners] = (GenericScannerFunc)regex_scanner_scan;
            active_configs[num_scanners++] = &core->regex_scanner_config;
            active_scanners[num_scanners] = (GenericScannerFunc)prompt_guard_scanner_scan;
            active_configs[num_scanners++] = &core->prompt_guard_config;
            break;
        case ROLE_ASSISTANT:
            active_scanners[num_scanners] = (GenericScannerFunc)regex_scanner_scan;
            active_configs[num_scanners++] = &core->regex_scanner_config;
            // Potentially add other scanners for assistant, e.g. a specialized CodeShield
            break;
        // Add other roles as needed
        default:
            break;
    }

    CScanResult final_result = {
        .decision = SCAN_DECISION_ALLOW,
        .reason = strdup("Default: No applicable scanners or all allowed"),
        .score = 0.0,
        .status = SCAN_STATUS_SUCCESS
    };
    int block_encountered = 0;

    for (int i = 0; i < num_scanners; ++i) {
        if (active_scanners[i]) {
            CScanResult scanner_result = active_scanners[i](input_message, input_trace, active_configs[i]);
            printf("Scanner %d result: %s, score: %.2f, reason: %s\n",
                   i, c_scan_decision_to_string(scanner_result.decision), scanner_result.score, scanner_result.reason);

            if (scanner_result.decision == SCAN_DECISION_BLOCK) {
                free_scan_result_members(&final_result); // Free previous reason
                final_result = scanner_result; // Overwrite with block
                block_encountered = 1;
                break; // Block from one scanner is enough
            } else if (scanner_result.decision == SCAN_DECISION_HUMAN_IN_THE_LOOP_REQUIRED) {
                // Handle HITL - for now, treat similar to a high-scoring allow or a soft block
                // If not already blocked, and this score is higher, update.
                if (!block_encountered && scanner_result.score > final_result.score) {
                    free_scan_result_members(&final_result);
                    final_result = scanner_result;
                } else {
                    free_scan_result_members(&scanner_result);
                }
            } else if (scanner_result.decision == SCAN_DECISION_ALLOW) {
                if (!block_encountered && scanner_result.score > final_result.score) {
                    // If current final decision is ALLOW, take the one with higher score (less likely to be problematic)
                    // Or, if we want to aggregate reasons, this logic needs to be more complex.
                    // For now, just take the highest score if all are allow.
                    // This part of the logic from Python (aggregating reasons, highest score) needs careful translation.
                    // The current Python code: "Select BLOCK as the final decision if present ... otherwise select the decision with the highest score"
                    // For non-block, this means we'd track the highest score.
                    // Let's simplify: if not blocked, current highest score for non-BLOCK results.
                     free_scan_result_members(&final_result);
                     final_result = scanner_result;
                } else {
                     free_scan_result_members(&scanner_result);
                }
            } else { // ERROR
                free_scan_result_members(&scanner_result); // Discard error results for now or handle them
            }
        }
    }

    // If no scanners were run, or all allowed but final_result reason is still default.
    if (num_scanners == 0 && final_result.decision == SCAN_DECISION_ALLOW && strcmp(final_result.reason, "Default: No applicable scanners or all allowed") == 0) {
        free(final_result.reason);
        final_result.reason = strdup("No scanners configured for this role.");
    }


    printf("Final decision: %s, Score: %.2f, Reason: %s\n",
           c_scan_decision_to_string(final_result.decision),
           final_result.score,
           final_result.reason);

    return final_result;
}

// Implementations for c_..._to_string and c_..._from_string are now in llamafirewall_types.c
