#!/bin/bash
# Comprehensive test script for differential-debugging-deps package
# Tests all critical functionality needed for AutoPatch benchmark

echo "=========================================="
echo "Testing differential-debugging-deps package"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

test_pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((pass_count++))
}

test_fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    echo -e "${RED}  Error: $2${NC}"
    ((fail_count++))
}

test_warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

echo "=== System Information ==="
cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID"
echo ""

echo "=== Test 1: Python 3.7 Installation ==="
if [ -x /usr/local/python37/bin/python3.7 ]; then
    VERSION=$(/usr/local/python37/bin/python3.7 --version 2>&1)
    if [[ "$VERSION" == "Python 3.7.17" ]]; then
        test_pass "Python 3.7.17 installed at /usr/local/python37/bin/python3.7"
    else
        test_fail "Python version incorrect" "Got: $VERSION"
    fi
else
    test_fail "Python 3.7 executable not found" "/usr/local/python37/bin/python3.7 does not exist"
fi

echo ""
echo "=== Test 2: Python 3.7 Symlink ==="
if command -v python3.7 >/dev/null 2>&1; then
    VERSION=$(python3.7 --version 2>&1)
    if [[ "$VERSION" == "Python 3.7.17" ]]; then
        test_pass "python3.7 symlink works"
    else
        test_fail "python3.7 symlink version incorrect" "Got: $VERSION"
    fi
else
    test_fail "python3.7 not in PATH" "Symlink may not be created"
fi

echo ""
echo "=== Test 3: Python Shared Library ==="
if python3.7 -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    test_pass "Python shared library loads correctly"
else
    test_fail "Python shared library error" "Failed to import sys module"
fi

echo ""
echo "=== Test 4: LLDB Installation ==="
if [ -x /usr/local/lldb13/bin/lldb ]; then
    test_pass "LLDB binary installed at /usr/local/lldb13/bin/lldb"
else
    test_fail "LLDB binary not found" "/usr/local/lldb13/bin/lldb does not exist"
fi

echo ""
echo "=== Test 5: LLDB Version ==="
if LLDB_VERSION=$(/usr/local/lldb13/bin/lldb --version 2>&1 | head -1); then
    if [[ "$LLDB_VERSION" == *"13.0.1"* ]]; then
        test_pass "LLDB version 13.0.1"
        echo "  Full version: $LLDB_VERSION"
    else
        test_fail "LLDB version incorrect" "Got: $LLDB_VERSION"
    fi
else
    test_fail "LLDB --version failed" "Could not execute lldb"
fi

echo ""
echo "=== Test 6: LLDB Symlink ==="
if command -v lldb >/dev/null 2>&1; then
    test_pass "lldb symlink works"
else
    test_fail "lldb not in PATH" "Symlink may not be created"
fi

echo ""
echo "=== Test 7: LLDB Dependencies (libstdc++.so.6) ==="
if ldd /usr/local/lldb13/bin/lldb 2>&1 | grep -q "not found"; then
    MISSING=$(ldd /usr/local/lldb13/bin/lldb 2>&1 | grep "not found")
    test_fail "LLDB has missing dependencies" "$MISSING"
else
    test_pass "All LLDB shared libraries found"
fi

echo ""
echo "=== Test 8: GLIBCXX Version Check ==="
if strings /usr/local/lldb13/lib/libstdc++.so.6 2>/dev/null | grep -q "GLIBCXX_3.4.22"; then
    test_pass "libstdc++.so.6 includes GLIBCXX_3.4.22+"
    echo "  Available versions:"
    strings /usr/local/lldb13/lib/libstdc++.so.6 | grep GLIBCXX | sort -V | tail -3 | sed 's/^/  /'
else
    test_warn "Could not verify GLIBCXX version in bundled libstdc++.so.6"
fi

echo ""
echo "=== Test 9: Python Library Path ==="
if echo $LD_LIBRARY_PATH | grep -q "/usr/local/python37/lib"; then
    test_pass "LD_LIBRARY_PATH includes Python library directory"
else
    test_warn "LD_LIBRARY_PATH may not be set (sourcing /etc/profile.d/lldb-python.sh may be required)"
fi

echo ""
echo "=== Test 10: LLDB Library Path ==="
if echo $LD_LIBRARY_PATH | grep -q "/usr/local/lldb13/lib"; then
    test_pass "LD_LIBRARY_PATH includes LLDB library directory"
else
    test_warn "LD_LIBRARY_PATH may not be set (sourcing /etc/profile.d/lldb-python.sh may be required)"
fi

echo ""
echo "=== Test 11: LLDB Python Bindings Location ==="
if [ -d /usr/local/lldb13/lib/python3.7/site-packages/lldb ]; then
    test_pass "LLDB Python bindings directory exists"
    echo "  Location: /usr/local/lldb13/lib/python3.7/site-packages/lldb"
else
    test_fail "LLDB Python bindings directory not found" "Missing: /usr/local/lldb13/lib/python3.7/site-packages/lldb"
fi

echo ""
echo "=== Test 12: LLDB Python Module Files ==="
if [ -f /usr/local/lldb13/lib/python3.7/site-packages/lldb/__init__.py ]; then
    test_pass "LLDB __init__.py exists"
else
    test_fail "LLDB __init__.py missing" "Required for Python import"
fi

if [ -f /usr/local/lldb13/lib/python3.7/site-packages/lldb/_lldb.so ]; then
    test_pass "LLDB _lldb.so (SWIG binary) exists"
    FILE_INFO=$(file /usr/local/lldb13/lib/python3.7/site-packages/lldb/_lldb.so)
    echo "  Type: $FILE_INFO"
else
    test_fail "LLDB _lldb.so missing" "SWIG-generated binary not found"
fi

echo ""
echo "=== Test 13: PYTHONPATH Configuration ==="
if echo $PYTHONPATH | grep -q "/usr/local/lldb13/lib/python3.7/site-packages"; then
    test_pass "PYTHONPATH includes LLDB Python bindings"
else
    test_warn "PYTHONPATH may not be set (sourcing /etc/profile.d/lldb-python.sh may be required)"
    # Set it for remaining tests
    export PYTHONPATH="/usr/local/lldb13/lib/python3.7/site-packages:$PYTHONPATH"
fi

echo ""
echo "=== Test 14: Environment Setup Script ==="
if [ -f /etc/profile.d/lldb-python.sh ]; then
    test_pass "Environment setup script exists"
    echo "  Sourcing /etc/profile.d/lldb-python.sh..."
    source /etc/profile.d/lldb-python.sh
else
    test_fail "Environment setup script missing" "/etc/profile.d/lldb-python.sh not found"
fi

echo ""
echo "=== Test 15: CRITICAL - Import LLDB Module ==="
if python3.7 -c "import lldb" 2>/dev/null; then
    test_pass "Python can import lldb module"
else
    ERROR=$(python3.7 -c "import lldb" 2>&1 || true)
    test_fail "Cannot import lldb module" "$ERROR"
fi

echo ""
echo "=== Test 16: CRITICAL - LLDB Version via Python ==="
if VERSION=$(python3.7 -c "import lldb; print(lldb.SBDebugger.GetVersionString())" 2>&1); then
    if [[ "$VERSION" == *"13.0.1"* ]] || [[ "$VERSION" == *"version 13"* ]]; then
        test_pass "LLDB Python bindings return correct version"
        echo "  Version: $VERSION"
    else
        test_fail "LLDB Python version unexpected" "Got: $VERSION"
    fi
else
    test_fail "Cannot get LLDB version from Python" "$VERSION"
fi

echo ""
echo "=== Test 17: CRITICAL - LLDB Python API ==="
if python3.7 -c "
import lldb
debugger = lldb.SBDebugger.Create()
assert debugger.IsValid(), 'Debugger not valid'
target = debugger.CreateTarget('/bin/ls')
assert target.IsValid(), 'Target not valid'
lldb.SBDebugger.Terminate()
print('SUCCESS')
" 2>/dev/null | grep -q "SUCCESS"; then
    test_pass "LLDB Python API fully functional"
else
    ERROR=$(python3.7 -c "
import lldb
debugger = lldb.SBDebugger.Create()
assert debugger.IsValid(), 'Debugger not valid'
target = debugger.CreateTarget('/bin/ls')
assert target.IsValid(), 'Target not valid'
lldb.SBDebugger.Terminate()
print('SUCCESS')
" 2>&1 || true)
    test_fail "LLDB Python API test failed" "$ERROR"
fi

echo ""
echo "=== Test 18: LLDB Module Location ==="
if MODULE_PATH=$(python3.7 -c "import lldb; print(lldb.__file__)" 2>&1); then
    if [[ "$MODULE_PATH" == *"/usr/local/lldb13/lib/python3.7/site-packages/lldb/"* ]]; then
        test_pass "LLDB module loaded from correct location"
        echo "  Path: $MODULE_PATH"
    else
        test_warn "LLDB module loaded from unexpected location: $MODULE_PATH"
    fi
else
    test_fail "Cannot determine LLDB module location" "$MODULE_PATH"
fi

echo ""
echo "=== Test 19: LLDB SWIG Binary Dependencies ==="
if [ -f /usr/local/lldb13/lib/python3.7/site-packages/lldb/_lldb.so ]; then
    if ldd /usr/local/lldb13/lib/python3.7/site-packages/lldb/_lldb.so 2>&1 | grep -q "not found"; then
        MISSING=$(ldd /usr/local/lldb13/lib/python3.7/site-packages/lldb/_lldb.so 2>&1 | grep "not found")
        test_fail "_lldb.so has missing dependencies" "$MISSING"
    else
        test_pass "All _lldb.so dependencies satisfied"
    fi
else
    test_warn "_lldb.so not found, skipping dependency check"
fi

echo ""
echo "=== Test 20: ldconfig Configuration ==="
if [ -f /etc/ld.so.conf.d/python37.conf ]; then
    test_pass "Python ldconfig entry exists"
    echo "  Content: $(cat /etc/ld.so.conf.d/python37.conf)"
else
    test_fail "Python ldconfig entry missing" "/etc/ld.so.conf.d/python37.conf not found"
fi

if [ -f /etc/ld.so.conf.d/lldb13.conf ]; then
    test_pass "LLDB ldconfig entry exists"
    echo "  Content: $(cat /etc/ld.so.conf.d/lldb13.conf)"
else
    test_fail "LLDB ldconfig entry missing" "/etc/ld.so.conf.d/lldb13.conf not found"
fi

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo "The package is ready for AutoPatch benchmark!"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo "Please review failures above."
    exit 1
fi
