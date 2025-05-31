#ifndef PROMPT_GUARD_SCANNER_H
#define PROMPT_GUARD_SCANNER_H

#include "../llamafirewall_types.h"

typedef struct {
    // Configuration for PromptGuard, e.g., path to Python helper script, timeout
    char* python_script_path; // Example: "/opt/llamafirewall/promptguard_helper.py"
    double block_threshold;
    int initialized;
} PromptGuardScannerConfig;

// Initializes the PromptGuardScanner (e.g., checks if script exists)
int prompt_guard_scanner_init(PromptGuardScannerConfig* config, const char* script_path, double threshold);

// Frees resources
void prompt_guard_scanner_cleanup(PromptGuardScannerConfig* config);

// Scan function for PromptGuardScanner
CScanResult prompt_guard_scanner_scan(
    const PromptGuardScannerConfig* config,
    const CMessage* message,
    const CTrace* trace
);

#endif // PROMPT_GUARD_SCANNER_H
