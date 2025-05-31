#ifndef LLAMAFIREWALL_CORE_H
#define LLAMAFIREWALL_CORE_H

#include "llamafirewall_types.h"
#include "scanners/regex_scanner.h" // Include RegexScanner's header
#include "scanners/prompt_guard_scanner.h" // Include PromptGuardScanner's header

// Forward declaration for the scanner structure/interface
// This might become a more concrete structure later for a scanner registry
struct CScannerFuncs;

typedef struct {
    // For now, configuration is implicit.
    // Later, this could hold handles to loaded scanners,
    // configurations for which scanners to run per role, etc.
    // Example: Map<CRole, CScannerFuncs**> role_scanners;
    int initialized; // Simple flag to indicate if init was successful

    // Configuration for the regex scanner
    RegexScannerConfig regex_scanner_config;
    // Configuration for the prompt guard scanner
    PromptGuardScannerConfig prompt_guard_config;
} LlamaFirewallCore;

// Initializes the LlamaFirewallCore instance.
// Returns 0 on success, -1 on error.
int llamafirewall_core_init(LlamaFirewallCore* core);

// Frees any resources associated with the LlamaFirewallCore instance.
void llamafirewall_core_cleanup(LlamaFirewallCore* core);

// Main scanning function.
// Caller is responsible for allocating memory for input_message and input_trace,
// and for freeing the returned CScanResult (specifically its char* members).
CScanResult llamafirewall_scan_message(
    LlamaFirewallCore* core,
    const CMessage* input_message,
    const CTrace* input_trace
);

// Placeholder for individual scanner function signature
// Each scanner implementation will provide a function matching this.
// This might be used if we have an array of generic scanner function pointers.
typedef CScanResult (*GenericScannerFunc)(const CMessage* message, const CTrace* trace, void* scanner_specific_config);


#endif // LLAMAFIREWALL_CORE_H
