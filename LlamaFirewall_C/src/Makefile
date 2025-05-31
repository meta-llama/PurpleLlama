CC = gcc
CFLAGS = -Wall -O2 -g
CFLAGS += $(shell pkg-config --cflags pcre2-8 systemd)
LDFLAGS = $(shell pkg-config --libs pcre2-8 systemd)
TARGET_EXEC = llamafirewall-dbus-service
SRCS = llamafirewall_types.c scanners/regex_scanner.c scanners/prompt_guard_scanner.c llamafirewall_core.c llamafirewall_dbus_service.c
OBJS = $(SRCS:.c=.o)
all: $(TARGET_EXEC)
$(TARGET_EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
clean:
	rm -f $(OBJS) $(TARGET_EXEC)
.PHONY: all clean

# --- Test Targets ---
TEST_DIR = ../tests
TEST_HELPERS = $(TEST_DIR)/test_helpers.h
# Common objects for linking tests if needed (e.g. llamafirewall_types.o)
COMMON_TEST_OBJS = llamafirewall_types.o

# Test for llamafirewall_types
TEST_TYPES_SRC = $(TEST_DIR)/test_llamafirewall_types.c
TEST_TYPES_EXEC = $(TEST_DIR)/test_llamafirewall_types
$(TEST_TYPES_EXEC): $(TEST_TYPES_SRC) $(COMMON_TEST_OBJS) $(TEST_HELPERS)
	$(CC) $(CFLAGS) -o $@ $(TEST_TYPES_SRC) $(COMMON_TEST_OBJS) $(LDFLAGS)

# Test for regex_scanner
TEST_REGEX_SCANNER_SRC = $(TEST_DIR)/test_regex_scanner.c
TEST_REGEX_SCANNER_OBJS = scanners/regex_scanner.o $(COMMON_TEST_OBJS)
TEST_REGEX_SCANNER_EXEC = $(TEST_DIR)/test_regex_scanner
$(TEST_REGEX_SCANNER_EXEC): $(TEST_REGEX_SCANNER_SRC) $(TEST_REGEX_SCANNER_OBJS) $(TEST_HELPERS)
	$(CC) $(CFLAGS) -o $@ $(TEST_REGEX_SCANNER_SRC) $(TEST_REGEX_SCANNER_OBJS) $(LDFLAGS)


# Main test target
test: $(TEST_TYPES_EXEC) $(TEST_REGEX_SCANNER_EXEC)
	@echo "Running LlamaFirewall Types tests..."
	@$(TEST_TYPES_EXEC)
	@echo "\nRunning RegexScanner tests..."
	@$(TEST_REGEX_SCANNER_EXEC)
	@echo "\nC Unit tests finished."

# Add test executables to clean target
clean:
	rm -f $(OBJS) $(TARGET_EXEC) $(TEST_TYPES_EXEC) $(TEST_REGEX_SCANNER_EXEC) $(TEST_DIR)/*.o
