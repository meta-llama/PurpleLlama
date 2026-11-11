# Differential Debugging Dependencies Build

This directory contains scripts to build `.deb` packages with LLDB 13 and Python 3.7 for the AutoPatch benchmark's differential debugging functionality.

## Overview

The packages provide:
- **Python 3.7.17** (built from source for GLIBC compatibility)
- **LLDB 13.0.1** (built from source with Python bindings)
- Required libraries and environment configuration

Built for:
- Ubuntu 16.04 (GLIBC 2.23)
- Ubuntu 20.04 (GLIBC 2.31)

## Building the Packages

### Prerequisites
- Root/sudo access
- `podman` or `docker`
- Internet connection (~164MB download for sources)
- Disk space: ~10GB for build artifacts
- Time: 2-3 hours for cold start build

### Build Command

```bash
sudo bash build_deb_packages.sh
```

The script will:
1. Check for existing packages and prompt to rebuild
2. Download LLVM 13.0.1 source (~141MB) and Python 3.7.17 source (~23MB)
3. Build Python 3.7 for each Ubuntu version (~10-15 min each)
4. Build LLDB 13 with Python bindings (~90-120 min each)
5. Create `.deb` packages in the current directory
6. Cache builds in `.build-cache/` for faster rebuilds

### Output Files

- `differential-debugging-deps-16.04.deb` (~1.1GB)
- `differential-debugging-deps-20.04.deb` (~615MB)

### Cache Management

The script caches builds in `.build-cache/` to speed up rebuilds:

```bash
# View cache
ls -lh .build-cache/

# Clear cache to force full rebuild
rm -rf .build-cache/

# Clear temporary build files
sudo rm -rf /tmp/deb-packaging
```

## Testing the Packages

### Test Ubuntu 16.04 Package

Start a test container with the arvo vulnerable image:

```bash
cd /mnt/PurpleLlama/CybersecurityBenchmarks/benchmark/autopatch/build

podman run -d --name arvo-test \
  -v "$(pwd)/differential-debugging-deps-16.04.deb:/tmp/deps.deb:ro" \
  -v "$(pwd)/test_deb_install.sh:/tmp/test.sh:ro" \
  docker.io/n132/arvo:12803-vul \
  sleep infinity
```

Install the package:

```bash
podman exec arvo-test bash -c "apt-get update -qq && apt-get install -y /tmp/deps.deb"
```

Run comprehensive tests:

```bash
podman exec arvo-test bash /tmp/test.sh
```

Clean up:

```bash
podman stop arvo-test && podman rm arvo-test
```

### Test Ubuntu 20.04 Package

Start a test container with vanilla Ubuntu 20.04:

```bash
podman run -d --name ubuntu20-test \
  -v "$(pwd)/differential-debugging-deps-20.04.deb:/tmp/deps.deb:ro" \
  -v "$(pwd)/test_deb_install.sh:/tmp/test.sh:ro" \
  docker.io/library/ubuntu:20.04 \
  sleep infinity
```

Install the package and dependencies:

```bash
podman exec ubuntu20-test bash -c "apt-get update -qq && apt-get install -y /tmp/deps.deb"

# Ubuntu 20.04 requires additional runtime dependencies
podman exec ubuntu20-test apt-get install -y libxml2 libedit2
```

Run tests:

```bash
podman exec ubuntu20-test bash /tmp/test.sh
```

Clean up:

```bash
podman stop ubuntu20-test && podman rm ubuntu20-test
```

## Test Coverage

The `test_deb_install.sh` script performs 20 comprehensive tests:

1. Python 3.7 installation and version
2. Python symlink (`python3.7` in PATH)
3. Python shared library loading
4. LLDB installation and version
5. LLDB symlink (`lldb` in PATH)
6. LLDB dependencies check
7. GLIBCXX version verification
8. Library path configuration
9. LLDB Python bindings location
10. LLDB Python module files
11. Environment setup script
12. **CRITICAL:** Python can import `lldb` module
13. **CRITICAL:** LLDB version via Python API
14. **CRITICAL:** LLDB Python API functionality
15. LLDB module location verification
16. LLDB SWIG binary dependencies
17. ldconfig configuration

All tests must pass for the package to be ready for AutoPatch benchmark.

## Known Issues

### Ubuntu 20.04 Runtime Dependencies

The Ubuntu 20.04 package requires system libraries that are not bundled:
- `libxml2` (libxml2.so.2)
- `libedit2` (libedit.so.2)

Install with: `apt-get install -y libxml2 libedit2`

The Ubuntu 16.04 package is fully self-contained (these libraries are typically pre-installed in arvo containers).

## Troubleshooting

### PPA Service Unavailable (503 Error)

If you encounter "Service Unavailable" when adding the ubuntu-toolchain-r PPA:
- The script automatically retries up to 3 times with 10-second delays
- Launchpad service may be temporarily down
- Wait a few minutes and retry the build

## Files

- `build_deb_packages.sh` - Main build script
- `test_deb_install.sh` - Comprehensive test suite
- `differential-debugging-deps-16.04.deb` - Ubuntu 16.04 package (generated)
- `differential-debugging-deps-20.04.deb` - Ubuntu 20.04 package (generated)
- `.build-cache/` - Cached builds (generated)
