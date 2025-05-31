#ifndef REGEX_SCANNER_H
#define REGEX_SCANNER_H

#include "../llamafirewall_types.h" // For CMessage, CTrace, CScanResult
#include <stddef.h> // For size_t

// Define a structure to hold the RegexScanner's state, including compiled patterns.
// We will use PCRE2 for regex compilation and matching.
// The actual pcre2_code type will be included in the .c file.
typedef struct RegexPattern {
    char* name;
    void* compiled_pattern; // To be cast to pcre2_code*
} RegexPattern;

typedef struct {
    RegexPattern* patterns;
    size_t num_patterns;
    double block_threshold; // Though for regex, it's usually a direct match/block
} RegexScannerConfig;

// Initializes the RegexScanner, compiles patterns.
// Returns 0 on success, -1 on error.
int regex_scanner_init(RegexScannerConfig* config);

// Frees resources used by the RegexScanner (compiled patterns).
void regex_scanner_cleanup(RegexScannerConfig* config);

// Scan function for the RegexScanner.
CScanResult regex_scanner_scan(
    const RegexScannerConfig* config,
    const CMessage* message,
    const CTrace* trace // trace might not be used by this specific scanner
);

#endif // REGEX_SCANNER_H
