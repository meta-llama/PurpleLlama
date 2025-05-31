#ifndef LLAMAFIREWALL_TYPES_H
#define LLAMAFIREWALL_TYPES_H

#include <stddef.h> // For size_t
#include <stdint.h> // For fixed-width integers if needed

// From Python: class ScannerType(Enum):
typedef enum {
    SCANNER_TYPE_CODE_SHIELD,
    SCANNER_TYPE_PROMPT_GUARD,
    SCANNER_TYPE_AGENT_ALIGNMENT,
    SCANNER_TYPE_HIDDEN_ASCII,
    SCANNER_TYPE_PII_DETECTION,
    SCANNER_TYPE_REGEX,
    SCANNER_TYPE_UNKNOWN // For error or default cases
} CScannerType;

// From Python: class ScanDecision(Enum):
typedef enum {
    SCAN_DECISION_ALLOW,
    SCAN_DECISION_HUMAN_IN_THE_LOOP_REQUIRED,
    SCAN_DECISION_BLOCK,
    SCAN_DECISION_ERROR // For error cases
} CScanDecision;

// From Python: class ScanStatus(Enum):
typedef enum {
    SCAN_STATUS_SUCCESS,
    SCAN_STATUS_ERROR,
    SCAN_STATUS_SKIPPED
} CScanStatus;

// From Python: class Role(Enum):
typedef enum {
    ROLE_TOOL,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_MEMORY,
    ROLE_SYSTEM,
    ROLE_UNKNOWN // For error or default cases
} CRole;

// From Python: @dataclass class Message:
typedef struct {
    CRole role;
    char* content; // Dynamically allocated string
    // tool_calls: Optional[List[Dict[str, Any]]] - This is complex for C.
    // For simplicity in the initial C version, we might omit tool_calls
    // or represent it as a JSON string if essential.
    // For now, omitting it.
} CMessage;

// From Python: Trace: TypeAlias = List[Message]
typedef struct {
    CMessage* messages;
    size_t count;
} CTrace;

// From Python: @dataclass class ScanResult:
typedef struct {
    CScanDecision decision;
    char* reason; // Dynamically allocated string
    double score;
    CScanStatus status;
} CScanResult;

// Helper functions for string to enum conversions (declarations)
// These will be implemented in a corresponding .c file
CScannerType c_scanner_type_from_string(const char* s);
const char* c_scanner_type_to_string(CScannerType type);

CScanDecision c_scan_decision_from_string(const char* s);
const char* c_scan_decision_to_string(CScanDecision decision);

CRole c_role_from_string(const char* s);
const char* c_role_to_string(CRole role);

CScanStatus c_scan_status_from_string(const char* s);
const char* c_scan_status_to_string(CScanStatus status);

#endif // LLAMAFIREWALL_TYPES_H
