#include "regex_scanner.h"
#include <pcre2.h> // PCRE2 library for regex
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default regex patterns (similar to the Python version)
// Using an array of structs for easier initialization
typedef struct {
    const char* name;
    const char* pattern;
} DefaultPattern;

static const DefaultPattern DEFAULT_REGEX_PATTERNS_DATA[] = {
    {"Prompt injection", "ignore previous instructions|ignore all instructions"},
    {"Email address", "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"},
    {"Phone number", "\\b(\\+\\d{1,2}\\s?)?\\(?\\d{3}\\)?[\\s.-]?\\d{3}[\\s.-]?\\d{4}\\b"},
    // {"Credit card", "\\b(?:\\d{4}[- ]?){3}\\d{4}\\b"}, // PCRE2 might need different syntax for non-capturing group, or just use capturing
    {"Credit card", "\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b"},
    {"Social security number", "\\b\\d{3}-\\d{2}-\\d{4}\\b"}
};
static const size_t NUM_DEFAULT_PATTERNS = sizeof(DEFAULT_REGEX_PATTERNS_DATA) / sizeof(DEFAULT_REGEX_PATTERNS_DATA[0]);

int regex_scanner_init(RegexScannerConfig* config) {
    if (!config) return -1;

    config->patterns = (RegexPattern*)malloc(NUM_DEFAULT_PATTERNS * sizeof(RegexPattern));
    if (!config->patterns) {
        perror("Failed to allocate memory for regex patterns");
        return -1;
    }
    config->num_patterns = 0; // Will be incremented as patterns are successfully compiled

    int error_code;
    PCRE2_SIZE error_offset;

    for (size_t i = 0; i < NUM_DEFAULT_PATTERNS; ++i) {
        pcre2_code* re = pcre2_compile(
            (PCRE2_SPTR)DEFAULT_REGEX_PATTERNS_DATA[i].pattern, // pattern
            PCRE2_ZERO_TERMINATED, // string is zero-terminated
            PCRE2_CASELESS | PCRE2_DOTALL, // options
            &error_code,      // for error code
            &error_offset,    // for error offset
            NULL              // use default compile context
        );

        if (re == NULL) {
            PCRE2_UCHAR buffer[256];
            pcre2_get_error_message(error_code, buffer, sizeof(buffer));
            fprintf(stderr, "PCRE2 compilation failed for '%s' at offset %d: %s\n",
                    DEFAULT_REGEX_PATTERNS_DATA[i].name, (int)error_offset, buffer);
            // Do not add this pattern, continue with others
            continue;
        }
        config->patterns[config->num_patterns].name = strdup(DEFAULT_REGEX_PATTERNS_DATA[i].name);
        config->patterns[config->num_patterns].compiled_pattern = re;
        config->num_patterns++;
    }

    // If no patterns compiled successfully, it might be an issue.
    if (config->num_patterns == 0 && NUM_DEFAULT_PATTERNS > 0) {
        fprintf(stderr, "RegexScanner: Warning - no regex patterns were successfully compiled.\n");
        // Depending on desired strictness, could return -1 here.
    }

    config->block_threshold = 1.0; // Standard for regex: a match is a block.
    printf("RegexScanner initialized with %zu patterns.\n", config->num_patterns);
    return 0;
}

void regex_scanner_cleanup(RegexScannerConfig* config) {
    if (!config || !config->patterns) return;

    for (size_t i = 0; i < config->num_patterns; ++i) {
        free(config->patterns[i].name);
        if (config->patterns[i].compiled_pattern) {
            pcre2_code_free((pcre2_code*)config->patterns[i].compiled_pattern);
        }
    }
    free(config->patterns);
    config->patterns = NULL;
    config->num_patterns = 0;
    printf("RegexScanner cleaned up.\n");
}

CScanResult regex_scanner_scan(
    const RegexScannerConfig* config,
    const CMessage* message,
    const CTrace* trace // Not used by this scanner
) {
    if (!config || !message || !message->content) {
        return (CScanResult){
            .decision = SCAN_DECISION_ERROR,
            .reason = strdup("RegexScanner: Invalid arguments or uninitialized"),
            .score = 0.0,
            .status = SCAN_STATUS_ERROR
        };
    }

    // Each pattern has its own pcre2_code, so match_data should be created per pattern,
    // or more efficiently, a single match_data created with pcre2_match_data_create
    // and large enough for any of the patterns.
    // For simplicity here, we'll create it inside the loop if needed, or pass NULL if the pattern
    // doesn't need to store captures (which these don't explicitly).
    // However, pcre2_match requires a match_data block.

    pcre2_match_data* match_data = NULL; // Will be created from the first valid pattern

    for (size_t i = 0; i < config->num_patterns; ++i) {
        if (!config->patterns[i].compiled_pattern) continue; // Skip if compilation failed

        if (match_data == NULL) { // Create match_data using the first available compiled pattern
             match_data = pcre2_match_data_create_from_pattern(
                (const pcre2_code*)config->patterns[i].compiled_pattern, NULL);
             if (match_data == NULL) {
                return (CScanResult){
                    .decision = SCAN_DECISION_ERROR,
                    .reason = strdup("RegexScanner: Failed to create match_data"),
                    .score = 0.0,
                    .status = SCAN_STATUS_ERROR};
            }
        }


        int rc = pcre2_match(
            (const pcre2_code*)config->patterns[i].compiled_pattern, // compiled pattern
            (PCRE2_SPTR)message->content, // subject string
            PCRE2_ZERO_TERMINATED, // length of subject
            0,                     // start offset in subject
            0,                     // options
            match_data,            // block for storing results
            NULL                   // use default match context
        );

        if (rc >= 0) { // Match found
            char* reason_str;
            int len = snprintf(NULL, 0, "Regex match: %s", config->patterns[i].name);
            reason_str = (char*)malloc(len + 1);
            if (reason_str) {
                 sprintf(reason_str, "Regex match: %s", config->patterns[i].name);
            } else {
                 reason_str = strdup("Regex match: <unknown pattern name - alloc error>"); // Fallback
            }

            if (match_data) pcre2_match_data_free(match_data);
            return (CScanResult){
                .decision = SCAN_DECISION_BLOCK,
                .reason = reason_str,
                .score = 1.0, // Full block score
                .status = SCAN_STATUS_SUCCESS
            };
        } else if (rc == PCRE2_ERROR_NOMATCH) {
            // No match, continue to next pattern
        } else {
            // Some other error
            PCRE2_UCHAR buffer[120];
            pcre2_get_error_message(rc, buffer, sizeof(buffer));
            fprintf(stderr, "RegexScanner: Match error for pattern '%s': %s\n", config->patterns[i].name, buffer);
            // Potentially return an error or just skip this pattern
        }
    }

    if (match_data) pcre2_match_data_free(match_data);
    return (CScanResult){
        .decision = SCAN_DECISION_ALLOW,
        .reason = strdup("RegexScanner: No patterns matched"),
        .score = 0.0,
        .status = SCAN_STATUS_SUCCESS
    };
}
