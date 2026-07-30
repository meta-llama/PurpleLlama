#!/bin/bash
# Build differential-debugging-deps .deb packages for AutoPatch benchmark
# Creates both Ubuntu 16.04 and 20.04 versions
# 
# CRITICAL: 
# - Pre-built Python binaries require GLIBC 2.34+ but Ubuntu 16.04 has 2.23 and 20.04 has 2.31
# - Pre-built LLDB releases don't include Python bindings
# - We MUST build both Python AND LLDB from source in target OS containers
#
# Build time: ~2-3 hours (Python: 10-15 min, LLDB: 90-120 min per OS)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/tmp/deb-packaging"
LLVM_VERSION="13.0.1"
PYTHON_VERSION="3.7.17"

# Cache directory for reusing builds
CACHE_DIR="${SCRIPT_DIR}/.build-cache"

DEB_16_04="${SCRIPT_DIR}/differential-debugging-deps-16.04.deb"
DEB_20_04="${SCRIPT_DIR}/differential-debugging-deps-20.04.deb"

BUILD_16_04=true
BUILD_20_04=true
REBUILD_PYTHON_16_04=true
REBUILD_PYTHON_20_04=true
REBUILD_LLDB_16_04=true
REBUILD_LLDB_20_04=true

# Check if .deb files already exist
if [ -f "$DEB_16_04" ]; then
    echo "Found existing package: $(basename "$DEB_16_04")"
    read -p "Rebuild 16.04 package? (120-180 min) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        BUILD_16_04=false
    fi
fi

if [ -f "$DEB_20_04" ]; then
    echo "Found existing package: $(basename "$DEB_20_04")"
    read -p "Rebuild 20.04 package? (120-180 min) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        BUILD_20_04=false
    fi
fi

if [ "$BUILD_16_04" = false ] && [ "$BUILD_20_04" = false ]; then
    echo "Using existing packages."
    exit 0
fi

# Check for cached builds
mkdir -p "$CACHE_DIR"
echo ""
echo "=== Checking for cached builds ==="

if [ -d "$CACHE_DIR/python37-ubuntu16.04" ]; then
    echo "Found cached Python 3.7 for Ubuntu 16.04"
    REBUILD_PYTHON_16_04=false
fi

if [ -d "$CACHE_DIR/python37-ubuntu20.04" ]; then
    echo "Found cached Python 3.7 for Ubuntu 20.04"
    REBUILD_PYTHON_20_04=false
fi

if [ -d "$CACHE_DIR/lldb13-ubuntu16.04" ]; then
    echo "Found cached LLDB 13 for Ubuntu 16.04 (saves ~90-120 min!)"
    REBUILD_LLDB_16_04=false
fi

if [ -d "$CACHE_DIR/lldb13-ubuntu20.04" ]; then
    echo "Found cached LLDB 13 for Ubuntu 20.04 (saves ~90-120 min!)"
    REBUILD_LLDB_20_04=false
fi

if [ "$REBUILD_LLDB_16_04" = false ] && [ "$REBUILD_LLDB_20_04" = false ]; then
    echo "All builds cached! Package creation will be fast (~2-3 min)"
fi

echo ""
echo "=== Installing build dependencies ==="
apt-get update -qq
apt-get install -y -qq wget curl podman

echo "=== Creating build environment ==="
# Don't delete shared directory - it contains cached sources
mkdir -p "$BUILD_DIR/shared"
rm -rf "$BUILD_DIR/python37-ubuntu16.04" "$BUILD_DIR/python37-ubuntu20.04"
rm -rf "$BUILD_DIR/lldb13-ubuntu16.04" "$BUILD_DIR/lldb13-ubuntu20.04"
rm -rf "$BUILD_DIR"/*.deb

# ==========================================
# Download LLVM source (shared, cached)
# ==========================================
cd "$BUILD_DIR/shared"

if [ -f "llvmorg-${LLVM_VERSION}.tar.gz" ]; then
    echo "=== Using cached LLVM ${LLVM_VERSION} source ($(ls -lh llvmorg-${LLVM_VERSION}.tar.gz | awk '{print $5}')) ==="
else
    echo "=== Downloading LLVM ${LLVM_VERSION} source (~3.5GB, ~10 min) ==="
    curl -L -o "llvmorg-${LLVM_VERSION}.tar.gz" \
        "https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-${LLVM_VERSION}.tar.gz"
    echo "LLVM source downloaded ($(ls -lh llvmorg-${LLVM_VERSION}.tar.gz | awk '{print $5}'))"
fi

# ==========================================
# Prepare Python source (shared, cached)
# ==========================================
if [ -f "Python-${PYTHON_VERSION}.tgz" ]; then
    echo "=== Using cached Python ${PYTHON_VERSION} source ($(ls -lh Python-${PYTHON_VERSION}.tgz | awk '{print $5}')) ==="
else
    echo "=== Downloading Python ${PYTHON_VERSION} source (~23MB) ==="
    curl -L -o "Python-${PYTHON_VERSION}.tgz" \
        "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
    echo "Python source downloaded ($(ls -lh Python-${PYTHON_VERSION}.tgz | awk '{print $5}'))"
fi

echo "Python source downloaded"

# ==========================================
# Extract libstdc++.so.6 from Ubuntu 20.04
# LLDB needs GLIBCXX_3.4.22+ but Ubuntu 16.04 only has 3.4.21
# ==========================================
echo "=== Extracting libstdc++.so.6 from Ubuntu 20.04 ==="
cd "$BUILD_DIR/shared"

podman pull docker.io/library/ubuntu:20.04 >/dev/null 2>&1
podman run --rm docker.io/library/ubuntu:20.04 cat /usr/lib/x86_64-linux-gnu/libstdc++.so.6 > libstdc++.so.6

echo "Extracted libstdc++.so.6 ($(ls -lh libstdc++.so.6 | awk '{print $5}'))"
echo "GLIBCXX versions available:"
strings libstdc++.so.6 | grep GLIBCXX | sort -V | tail -3

# ==========================================
# Build for Ubuntu 16.04
# ==========================================
if [ "$BUILD_16_04" = true ]; then
echo "=== Building for Ubuntu 16.04 (GLIBC 2.23) ==="
cd "$BUILD_DIR"

# Build or use cached Python
if [ "$REBUILD_PYTHON_16_04" = true ]; then
    echo "=== Step 1/2: Building Python 3.7 in Ubuntu 16.04 container (~10-15 min) ==="
    podman run --rm \
        -v "$BUILD_DIR/shared/Python-${PYTHON_VERSION}.tgz:/tmp/Python-${PYTHON_VERSION}.tgz:ro" \
        -v "$BUILD_DIR:/output" \
        docker.io/library/ubuntu:16.04 bash -c "
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq build-essential zlib1g-dev libssl-dev libffi-dev \
            libbz2-dev libreadline-dev libsqlite3-dev wget \
            libncurses5-dev libncursesw5-dev xz-utils tk-dev \
            libxml2-dev libxmlsec1-dev liblzma-dev
        
        cd /tmp
        tar -xzf Python-${PYTHON_VERSION}.tgz
        cd Python-${PYTHON_VERSION}
        
        echo 'Configuring Python...'
        ./configure --prefix=/usr/local/python37 \
            --with-ensurepip=install \
            --enable-shared \
            --disable-test-modules \
            --quiet
        
        echo \"Building Python with \$(nproc) cores...\"
        make -j\$(nproc) > /dev/null
        make install DESTDIR=/output/python37-ubuntu16.04 > /dev/null
        
        echo 'Python 3.7 built successfully'
    "
    
    # Cache the build
    cp -r "$BUILD_DIR/python37-ubuntu16.04" "$CACHE_DIR/"
    echo "✓ Cached Python build for future use"
else
    echo "=== Step 1/2: Using cached Python 3.7 for Ubuntu 16.04 ==="
    cp -r "$CACHE_DIR/python37-ubuntu16.04" "$BUILD_DIR/"
fi

PYTHON_16_04_DIR="$BUILD_DIR/python37-ubuntu16.04/usr/local/python37"
echo "✓ Python for 16.04 ready at: $PYTHON_16_04_DIR"

# Build or use cached LLDB
if [ "$REBUILD_LLDB_16_04" = true ]; then
    echo "=== Step 2/2: Building LLDB ${LLVM_VERSION} with Python bindings in Ubuntu 16.04 (~90-120 min) ==="
    
    # Pre-create output directory structure to avoid permission issues
    echo "Preparing output directory..."
    sudo mkdir -p "$BUILD_DIR/lldb13-ubuntu16.04/usr/local/"{lldb13,include,bin,lib,share}
    sudo chmod -R 777 "$BUILD_DIR/lldb13-ubuntu16.04"
    
podman run --rm \
    -v "$BUILD_DIR/shared/llvmorg-${LLVM_VERSION}.tar.gz:/tmp/llvm-source.tar.gz:ro" \
    -v "$BUILD_DIR/python37-ubuntu16.04/usr/local/python37:/usr/local/python37:ro" \
    -v "$BUILD_DIR:/output" \
    docker.io/library/ubuntu:16.04 bash -c "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    
    echo 'Installing LLDB build dependencies...'
    apt-get update -qq
    apt-get install -y -qq build-essential ninja-build swig \
        libncurses5-dev libedit-dev libxml2-dev python3-dev \
        zlib1g-dev liblzma-dev wget curl libssl-dev software-properties-common
    
    # Install GCC 7 for LLDB 13 compatibility (GCC 5.4 in Ubuntu 16.04 is too old)
    echo 'Installing GCC 7 from ubuntu-toolchain-r PPA...'
    
    # Retry logic for PPA (Launchpad can be flaky)
    MAX_RETRIES=3
    RETRY_COUNT=0
    while [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; do
        if add-apt-repository ppa:ubuntu-toolchain-r/test -y; then
            echo 'PPA added successfully'
            break
        else
            RETRY_COUNT=\$((RETRY_COUNT + 1))
            if [ \$RETRY_COUNT -lt \$MAX_RETRIES ]; then
                echo \"PPA add failed, retrying (\$RETRY_COUNT/\$MAX_RETRIES) in 10 seconds...\"
                sleep 10
            else
                echo 'ERROR: Failed to add PPA after \$MAX_RETRIES attempts'
                echo 'Launchpad service may be temporarily unavailable'
                exit 1
            fi
        fi
    done
    
    apt-get update -qq
    apt-get install -y -qq gcc-7 g++-7
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
    echo \"GCC version: \$(gcc --version | head -1)\"
    
    # Build CMake 3.20.0 (Ubuntu 16.04 has 3.5.1, LLVM 13 requires 3.13.4+)
    echo 'Building CMake 3.20.0 (~10-15 min)...'
    cd /tmp
    wget -q https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
    tar -xzf cmake-3.20.0.tar.gz
    cd cmake-3.20.0
    ./bootstrap --parallel=\$(nproc) -- -DCMAKE_BUILD_TYPE=Release
    make -j\$(nproc)
    echo 'Installing CMake (output suppressed)...'
    make install > /dev/null 2>&1
    rm -rf /tmp/cmake-3.20.0 /tmp/cmake-3.20.0.tar.gz
    echo \"CMake version: \$(/usr/local/bin/cmake --version | head -1)\"
    
    # Ensure we use the newly installed cmake
    export PATH=/usr/local/bin:\$PATH
    
    cd /tmp
    echo 'Extracting LLVM source (~3.5GB)...'
    tar -xzf llvm-source.tar.gz
    cd llvm-project-llvmorg-${LLVM_VERSION}
    
    mkdir build
    cd build
    
    # Set library path so CMake can run Python interpreter
    export LD_LIBRARY_PATH=/usr/local/python37/lib:\$LD_LIBRARY_PATH
    
    # Verify Python works before configuring
    echo 'Testing Python 3.7...'
    /usr/local/python37/bin/python3.7 --version || echo 'WARNING: Python test failed'
    
    echo 'Configuring LLDB with CMake...'
    /usr/local/bin/cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS='clang;lldb' \
        -DCMAKE_INSTALL_PREFIX=/usr/local/lldb13 \
        -DLLDB_ENABLE_PYTHON=ON \
        -DPython3_EXECUTABLE=/usr/local/python37/bin/python3.7 \
        -DPython3_LIBRARY=/usr/local/python37/lib/libpython3.7m.so \
        -DPython3_INCLUDE_DIR=/usr/local/python37/include/python3.7m \
        -DPython3_ROOT_DIR=/usr/local/python37 \
        -DPYTHON_EXECUTABLE=/usr/local/python37/bin/python3.7 \
        -DPYTHON_LIBRARY=/usr/local/python37/lib/libpython3.7m.so \
        -DPYTHON_INCLUDE_DIR=/usr/local/python37/include/python3.7m \
        -DLLVM_TARGETS_TO_BUILD='X86' \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        > /dev/null
    
    echo \"Building LLDB (this takes 90-120 minutes with \$(nproc) cores)...\"
    echo \"Progress: Starting build at \$(date +%H:%M:%S)\"
    ninja lldb > /dev/null
    
    echo 'Installing LLDB...'
    DESTDIR=/output/lldb13-ubuntu16.04 ninja install > /dev/null 2>&1 || true
    
    # Manually copy lldb files if ninja install fails
    mkdir -p /output/lldb13-ubuntu16.04/usr/local/lldb13
    cp -r /tmp/llvm-project-llvmorg-${LLVM_VERSION}/build/lib /output/lldb13-ubuntu16.04/usr/local/lldb13/ 2>/dev/null || true
    cp -r /tmp/llvm-project-llvmorg-${LLVM_VERSION}/build/bin /output/lldb13-ubuntu16.04/usr/local/lldb13/ 2>/dev/null || true
    
    echo \"LLDB built successfully at \$(date +%H:%M:%S)\"
"
    
    # Cache the build
    cp -r "$BUILD_DIR/lldb13-ubuntu16.04" "$CACHE_DIR/"
    echo "✓ Cached LLDB build for future use"
else
    echo "=== Step 2/2: Using cached LLDB ${LLVM_VERSION} for Ubuntu 16.04 ==="
    cp -r "$CACHE_DIR/lldb13-ubuntu16.04" "$BUILD_DIR/"
fi

LLDB_16_04_DIR="$BUILD_DIR/lldb13-ubuntu16.04/usr/local/lldb13"
echo "✓ LLDB for 16.04 ready at: $LLDB_16_04_DIR"

echo "=== Packaging Ubuntu 16.04 ==="
cd "$BUILD_DIR"
mkdir -p 16.04/rootfs/usr/local

# Copy LLDB from build
echo "Installing LLDB ${LLVM_VERSION}..."
cp -r "$LLDB_16_04_DIR" 16.04/rootfs/usr/local/lldb13

# Copy newer libstdc++.so.6 for GLIBCXX_3.4.22+ support (Ubuntu 16.04 only has 3.4.21)
echo "Installing libstdc++.so.6 from Ubuntu 20.04..."
mkdir -p 16.04/rootfs/usr/local/lldb13/lib
cp "$BUILD_DIR/shared/libstdc++.so.6" 16.04/rootfs/usr/local/lldb13/lib/

# Install Python from Ubuntu 16.04 build
echo "Installing Python ${PYTHON_VERSION}..."
mkdir -p 16.04/rootfs/usr/local/python37
cp -r "$PYTHON_16_04_DIR"/* 16.04/rootfs/usr/local/python37/

# Create symlink for Python 3.7 in standard location (not python3, conflicts with system)
mkdir -p 16.04/rootfs/usr/bin
ln -sf /usr/local/python37/bin/python3.7 16.04/rootfs/usr/bin/python3.7

# Create LLDB symlink (already has LD_LIBRARY_PATH set via ldconfig)
ln -sf /usr/local/lldb13/bin/lldb 16.04/rootfs/usr/bin/lldb
ln -sf /usr/local/lldb13/bin/lldb 16.04/rootfs/usr/bin/lldb-13

# LLDB Python bindings should be built into lib/python3.7/site-packages/lldb
# Create symlink for Python to find them
mkdir -p 16.04/rootfs/usr/local/python37/lib/python3.7/site-packages
if [ -d "16.04/rootfs/usr/local/lldb13/lib/python3.7/site-packages/lldb" ]; then
    rm -f 16.04/rootfs/usr/local/python37/lib/python3.7/site-packages/lldb
    ln -s "/usr/local/lldb13/lib/python3.7/site-packages/lldb" \
        16.04/rootfs/usr/local/python37/lib/python3.7/site-packages/lldb
    echo "Created symlink for LLDB Python bindings"
else
    echo "Warning: LLDB Python bindings not found - checking build output"
    find 16.04/rootfs/usr/local/lldb13 -name "*lldb*.py" | head -5 || true
fi

# Create ldconfig entry for shared libraries
mkdir -p 16.04/rootfs/etc/ld.so.conf.d
echo "/usr/local/python37/lib" > 16.04/rootfs/etc/ld.so.conf.d/python37.conf
echo "/usr/local/lldb13/lib" >> 16.04/rootfs/etc/ld.so.conf.d/lldb13.conf

# Set up environment for LLDB Python bindings
mkdir -p 16.04/rootfs/etc/profile.d
cat > 16.04/rootfs/etc/profile.d/lldb-python.sh << 'ENVEOF'
export PYTHONPATH="/usr/local/lldb13/lib/python3.7/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/lldb13/lib:/usr/local/python37/lib:$LD_LIBRARY_PATH"
ENVEOF
chmod 644 16.04/rootfs/etc/profile.d/lldb-python.sh

# Create DEBIAN control directory
mkdir -p 16.04/rootfs/DEBIAN

cat > 16.04/rootfs/DEBIAN/control << 'EOF'
Package: differential-debugging-deps
Version: 1.0.0
Section: devel
Priority: optional
Architecture: amd64
Maintainer: PurpleLlama Community
Description: LLDB 13 and Python 3.7 for AutoPatch differential debugging
 This package contains LLDB 13 (built from source with Python 3.7 bindings)
 and Python 3.7 required for the AutoPatch benchmark differential debugging.
 Built specifically for Ubuntu 16.04 with GLIBC 2.23 compatibility.
EOF

# Create postinst script
cat > 16.04/rootfs/DEBIAN/postinst << 'EOF'
#!/bin/bash
set -e
ldconfig

# Non-fatal verification
if [ -x /usr/local/lldb13/bin/lldb ]; then
    echo "LLDB 13 installed at /usr/local/lldb13/bin/lldb"
else
    echo "Warning: LLDB binary not found"
fi

if [ -x /usr/local/python37/bin/python3.7 ]; then
    echo "Python 3.7 installed at /usr/local/python37/bin/python3.7"
else
    echo "Warning: Python 3.7 binary not found"
fi

exit 0
EOF
chmod 755 16.04/rootfs/DEBIAN/postinst

# Build the .deb package
cd 16.04
echo "Creating differential-debugging-deps-16.04.deb..."
dpkg-deb --root-owner-group --build rootfs differential-debugging-deps-16.04.deb

echo "✓ Created differential-debugging-deps-16.04.deb"
ls -lh differential-debugging-deps-16.04.deb

# Verify package
echo "Verifying package contents..."
dpkg-deb --info differential-debugging-deps-16.04.deb | grep -A5 "Package:"

# Copy to script directory
cp differential-debugging-deps-16.04.deb "$DEB_16_04"
echo "Copied to $(basename "$DEB_16_04")"

fi # End of BUILD_16_04

# ==========================================
# Build for Ubuntu 20.04
# ==========================================
if [ "$BUILD_20_04" = true ]; then
echo "=== Building for Ubuntu 20.04 (GLIBC 2.31) ==="
cd "$BUILD_DIR"

# Build or use cached Python
if [ "$REBUILD_PYTHON_20_04" = true ]; then
    echo "=== Step 1/2: Building Python 3.7 in Ubuntu 20.04 container (~10-15 min) ==="
    podman run --rm \
        -v "$BUILD_DIR/shared/Python-${PYTHON_VERSION}.tgz:/tmp/Python-${PYTHON_VERSION}.tgz:ro" \
        -v "$BUILD_DIR:/output" \
        docker.io/library/ubuntu:20.04 bash -c "
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq build-essential zlib1g-dev libssl-dev libffi-dev \
            libbz2-dev libreadline-dev libsqlite3-dev wget \
            libncurses5-dev libncursesw5-dev xz-utils tk-dev \
            libxml2-dev libxmlsec1-dev liblzma-dev
        
        cd /tmp
        tar -xzf Python-${PYTHON_VERSION}.tgz
        cd Python-${PYTHON_VERSION}
        
        echo 'Configuring Python...'
        ./configure --prefix=/usr/local/python37 \
            --with-ensurepip=install \
            --enable-shared \
            --disable-test-modules \
            --quiet
        
        echo \"Building Python with \$(nproc) cores...\"
        make -j\$(nproc) > /dev/null
        make install DESTDIR=/output/python37-ubuntu20.04 > /dev/null
        
        echo 'Python 3.7 built successfully'
    "
    
    # Cache the build
    cp -r "$BUILD_DIR/python37-ubuntu20.04" "$CACHE_DIR/"
    echo "✓ Cached Python build for future use"
else
    echo "=== Step 1/2: Using cached Python 3.7 for Ubuntu 20.04 ==="
    cp -r "$CACHE_DIR/python37-ubuntu20.04" "$BUILD_DIR/"
fi

PYTHON_20_04_DIR="$BUILD_DIR/python37-ubuntu20.04/usr/local/python37"
echo "✓ Python for 20.04 ready at: $PYTHON_20_04_DIR"

# Build or use cached LLDB
if [ "$REBUILD_LLDB_20_04" = true ]; then
    echo "=== Step 2/2: Building LLDB ${LLVM_VERSION} with Python bindings in Ubuntu 20.04 (~90-120 min) ==="
    
    # Pre-create output directory structure to avoid permission issues
    echo "Preparing output directory..."
    sudo mkdir -p "$BUILD_DIR/lldb13-ubuntu20.04/usr/local/"{lldb13,include,bin,lib,share}
    sudo chmod -R 777 "$BUILD_DIR/lldb13-ubuntu20.04"
    
podman run --rm \
    -v "$BUILD_DIR/shared/llvmorg-${LLVM_VERSION}.tar.gz:/tmp/llvm-source.tar.gz:ro" \
    -v "$BUILD_DIR/python37-ubuntu20.04/usr/local/python37:/usr/local/python37:ro" \
    -v "$BUILD_DIR:/output" \
    docker.io/library/ubuntu:20.04 bash -c "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    
    echo 'Installing LLDB build dependencies...'
    apt-get update -qq
    apt-get install -y -qq build-essential cmake ninja-build swig \
        libncurses5-dev libedit-dev libxml2-dev python3-dev \
        zlib1g-dev liblzma-dev wget curl
    
    cd /tmp
    echo 'Extracting LLVM source (~3.5GB)...'
    tar -xzf llvm-source.tar.gz
    cd llvm-project-llvmorg-${LLVM_VERSION}
    
    mkdir build
    cd build
    
    # Set library path so CMake can run Python interpreter
    export LD_LIBRARY_PATH=/usr/local/python37/lib:\$LD_LIBRARY_PATH
    
    # Verify Python works before configuring
    echo 'Testing Python 3.7...'
    /usr/local/python37/bin/python3.7 --version || echo 'WARNING: Python test failed'
    
    echo 'Configuring LLDB with CMake...'
    cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS='clang;lldb' \
        -DCMAKE_INSTALL_PREFIX=/usr/local/lldb13 \
        -DLLDB_ENABLE_PYTHON=ON \
        -DPython3_EXECUTABLE=/usr/local/python37/bin/python3.7 \
        -DPython3_LIBRARY=/usr/local/python37/lib/libpython3.7m.so \
        -DPython3_INCLUDE_DIR=/usr/local/python37/include/python3.7m \
        -DPython3_ROOT_DIR=/usr/local/python37 \
        -DPYTHON_EXECUTABLE=/usr/local/python37/bin/python3.7 \
        -DPYTHON_LIBRARY=/usr/local/python37/lib/libpython3.7m.so \
        -DPYTHON_INCLUDE_DIR=/usr/local/python37/include/python3.7m \
        -DLLVM_TARGETS_TO_BUILD='X86' \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        > /dev/null
    
    echo \"Building LLDB (this takes 90-120 minutes with \$(nproc) cores)...\"
    echo \"Progress: Starting build at \$(date +%H:%M:%S)\"
    ninja lldb > /dev/null
    
    echo 'Installing LLDB...'
    DESTDIR=/output/lldb13-ubuntu20.04 ninja install > /dev/null 2>&1 || true
    
    # Manually copy lldb files if ninja install fails
    mkdir -p /output/lldb13-ubuntu20.04/usr/local/lldb13
    cp -r /tmp/llvm-project-llvmorg-${LLVM_VERSION}/build/lib /output/lldb13-ubuntu20.04/usr/local/lldb13/ 2>/dev/null || true
    cp -r /tmp/llvm-project-llvmorg-${LLVM_VERSION}/build/bin /output/lldb13-ubuntu20.04/usr/local/lldb13/ 2>/dev/null || true
    
    echo \"LLDB built successfully at \$(date +%H:%M:%S)\"
"
    
    # Cache the build
    cp -r "$BUILD_DIR/lldb13-ubuntu20.04" "$CACHE_DIR/"
    echo "✓ Cached LLDB build for future use"
else
    echo "=== Step 2/2: Using cached LLDB ${LLVM_VERSION} for Ubuntu 20.04 ==="
    cp -r "$CACHE_DIR/lldb13-ubuntu20.04" "$BUILD_DIR/"
fi

LLDB_20_04_DIR="$BUILD_DIR/lldb13-ubuntu20.04/usr/local/lldb13"
echo "✓ LLDB for 20.04 ready at: $LLDB_20_04_DIR"

echo "=== Packaging Ubuntu 20.04 ==="
cd "$BUILD_DIR"
mkdir -p 20.04/rootfs/usr/local

# Copy LLDB from build
echo "Installing LLDB ${LLVM_VERSION}..."
cp -r "$LLDB_20_04_DIR" 20.04/rootfs/usr/local/lldb13

# Copy newer libstdc++.so.6 for consistency (Ubuntu 20.04 already has it but we include our own)
echo "Installing libstdc++.so.6 from Ubuntu 20.04..."
mkdir -p 20.04/rootfs/usr/local/lldb13/lib
cp "$BUILD_DIR/shared/libstdc++.so.6" 20.04/rootfs/usr/local/lldb13/lib/

# Install Python from Ubuntu 20.04 build
echo "Installing Python ${PYTHON_VERSION}..."
mkdir -p 20.04/rootfs/usr/local/python37
cp -r "$PYTHON_20_04_DIR"/* 20.04/rootfs/usr/local/python37/

# Create symlink for Python 3.7 in standard location (not python3, conflicts with system)
mkdir -p 20.04/rootfs/usr/bin
ln -sf /usr/local/python37/bin/python3.7 20.04/rootfs/usr/bin/python3.7

# Create LLDB symlink
ln -sf /usr/local/lldb13/bin/lldb 20.04/rootfs/usr/bin/lldb
ln -sf /usr/local/lldb13/bin/lldb 20.04/rootfs/usr/bin/lldb-13

# LLDB Python bindings
mkdir -p 20.04/rootfs/usr/local/python37/lib/python3.7/site-packages
if [ -d "20.04/rootfs/usr/local/lldb13/lib/python3.7/site-packages/lldb" ]; then
    rm -f 20.04/rootfs/usr/local/python37/lib/python3.7/site-packages/lldb
    ln -s "/usr/local/lldb13/lib/python3.7/site-packages/lldb" \
        20.04/rootfs/usr/local/python37/lib/python3.7/site-packages/lldb
    echo "Created symlink for LLDB Python bindings"
else
    echo "Warning: LLDB Python bindings not found - checking build output"
    find 20.04/rootfs/usr/local/lldb13 -name "*lldb*.py" | head -5 || true
fi

# Create ldconfig entry
mkdir -p 20.04/rootfs/etc/ld.so.conf.d
echo "/usr/local/python37/lib" > 20.04/rootfs/etc/ld.so.conf.d/python37.conf
echo "/usr/local/lldb13/lib" >> 20.04/rootfs/etc/ld.so.conf.d/lldb13.conf

# Set up environment for LLDB Python bindings
mkdir -p 20.04/rootfs/etc/profile.d
cat > 20.04/rootfs/etc/profile.d/lldb-python.sh << 'ENVEOF'
export PYTHONPATH="/usr/local/lldb13/lib/python3.7/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/lldb13/lib:/usr/local/python37/lib:$LD_LIBRARY_PATH"
ENVEOF
chmod 644 20.04/rootfs/etc/profile.d/lldb-python.sh

# Create DEBIAN control directory
mkdir -p 20.04/rootfs/DEBIAN

cat > 20.04/rootfs/DEBIAN/control << 'EOF'
Package: differential-debugging-deps
Version: 1.0.0
Section: devel
Priority: optional
Architecture: amd64
Maintainer: PurpleLlama Community
Description: LLDB 13 and Python 3.7 for AutoPatch differential debugging
 This package contains LLDB 13 (built from source with Python 3.7 bindings)
 and Python 3.7 required for the AutoPatch benchmark differential debugging.
 Built specifically for Ubuntu 20.04 with GLIBC 2.31 compatibility.
EOF

# Create postinst script
cat > 20.04/rootfs/DEBIAN/postinst << 'EOF'
#!/bin/bash
set -e
ldconfig

# Non-fatal verification
if [ -x /usr/local/lldb13/bin/lldb ]; then
    echo "LLDB 13 installed at /usr/local/lldb13/bin/lldb"
else
    echo "Warning: LLDB binary not found"
fi

if [ -x /usr/local/python37/bin/python3.7 ]; then
    echo "Python 3.7 installed at /usr/local/python37/bin/python3.7"
else
    echo "Warning: Python 3.7 binary not found"
fi

exit 0
EOF
chmod 755 20.04/rootfs/DEBIAN/postinst

# Build the .deb package
cd 20.04
echo "Creating differential-debugging-deps-20.04.deb..."
dpkg-deb --root-owner-group --build rootfs differential-debugging-deps-20.04.deb

echo "✓ Created differential-debugging-deps-20.04.deb"
ls -lh differential-debugging-deps-20.04.deb

# Verify package
echo "Verifying package contents..."
dpkg-deb --info differential-debugging-deps-20.04.deb | grep -A5 "Package:"

# Copy to script directory
cp differential-debugging-deps-20.04.deb "$DEB_20_04"
echo "Copied to $(basename "$DEB_20_04")"

fi # End of BUILD_20_04

# ==========================================
# Summary
# ==========================================
echo ""
echo "=== Build complete! ==="
echo "Packages available:"
if [ -f "$DEB_16_04" ]; then
    echo "  ✓ $(basename "$DEB_16_04")"
fi
if [ -f "$DEB_20_04" ]; then
    echo "  ✓ $(basename "$DEB_20_04")"
fi
echo ""
echo "Location: ${SCRIPT_DIR}/"
echo ""
echo "Build cache stored at: ${CACHE_DIR}/"
echo "  (Reusing cache saves ~90-120 min per rebuild!)"
echo ""
echo "To clear cache and rebuild from scratch:"
echo "  rm -rf ${CACHE_DIR}"
echo ""
echo "Test with:"
echo "  podman run --rm -v $DEB_16_04:/tmp/test.deb \\"
echo "    docker.io/n132/arvo:1236-vul bash -c \\"
echo "    'apt-get update -qq && apt-get install -y /tmp/test.deb && lldb --version && python3.7 --version && python3.7 -c \"import lldb; print(lldb.SBDebugger.GetVersionString())\"'"
