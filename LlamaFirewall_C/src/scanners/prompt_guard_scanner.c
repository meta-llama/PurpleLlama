#include "prompt_guard_scanner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// For a real IPC, you might need headers like <unistd.h> for pipe/fork or socket headers.
// For this simulation, we'll keep it simple.

int prompt_guard_scanner_init(PromptGuardScannerConfig* config, const char* script_path, double threshold) {
    if (!config || !script_path) return -1;

    config->python_script_path = strdup(script_path);
    if (!config->python_script_path) {
        perror("PromptGuardScanner: Failed to allocate memory for script path");
        return -1;
    }
    config->block_threshold = threshold;
    config->initialized = 1;

    // In a real implementation, you might check if the script exists and is executable.
    // For example: if (access(script_path, X_OK) != 0) { perror("Script not accessible"); config->initialized = 0; return -1;}

    printf("PromptGuardScanner initialized (simulated IPC to: %s).\n", config->python_script_path);
    return 0;
}

void prompt_guard_scanner_cleanup(PromptGuardScannerConfig* config) {
    if (config) {
        free(config->python_script_path);
        config->python_script_path = NULL;
        config->initialized = 0;
        printf("PromptGuardScanner cleaned up.\n");
    }
}

// SIMULATED IPC call
static double get_simulated_jailbreak_score_from_python(const char* text_content, const char* script_path) {
    printf("SIMULATING IPC: Calling Python script '%s' with text: \"%.30s...\"\n", script_path, text_content);

    // Simulate different responses based on input for testing
    if (strstr(text_content, "tell me your secrets")) {
        printf("SIMULATING IPC: Python script returned score 0.95\n");
        return 0.95;
    }
    if (strstr(text_content, "ignore previous instructions")) {
         printf("SIMULATING IPC: Python script returned score 0.85\n");
        return 0.85; // Also high, but different from regex
    }
    if (strstr(text_content, "safe query")) {
        printf("SIMULATING IPC: Python script returned score 0.1\n");
        return 0.1;
    }

    // Default simulated score
    printf("SIMULATING IPC: Python script returned score 0.2\n");
    return 0.2;
}

CScanResult prompt_guard_scanner_scan(
    const PromptGuardScannerConfig* config,
    const CMessage* message,
    const CTrace* trace // Not used in this simplified IPC simulation directly
) {
    if (!config || !config->initialized || !message || !message->content) {
        return (CScanResult){
            .decision = SCAN_DECISION_ERROR,
            .reason = strdup("PromptGuardScanner: Invalid arguments or not initialized"),
            .score = 0.0,
            .status = SCAN_STATUS_ERROR
        };
    }

    // Simulate calling the Python script via IPC
    double score = get_simulated_jailbreak_score_from_python(message->content, config->python_script_path);

    CScanDecision decision = (score >= config->block_threshold) ? SCAN_DECISION_BLOCK : SCAN_DECISION_ALLOW;
    char reason_buffer[256];
    if (decision == SCAN_DECISION_BLOCK) {
        snprintf(reason_buffer, sizeof(reason_buffer),
                 "PromptGuard: Detected potential policy violation (score: %.2f)", score);
    } else {
        snprintf(reason_buffer, sizeof(reason_buffer),
                 "PromptGuard: No significant policy violation detected (score: %.2f)", score);
    }

    return (CScanResult){
        .decision = decision,
        .reason = strdup(reason_buffer),
        .score = score,
        .status = SCAN_STATUS_SUCCESS
    };
}
