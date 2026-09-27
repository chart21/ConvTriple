#!/bin/bash

# CUDA architecture of the local GPU (e.g. 89 for Ada), overridable with GPU_ARCHITECTURE. Building
# for an older one makes the driver JIT-compile every kernel from PTX at startup.
GPU_ARCH=${GPU_ARCHITECTURE:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')}
GPU_ARCH=${GPU_ARCH:-"75;80;86;89"}

WORK_DIR="$PWD"
DEPS="$WORK_DIR/deps"

TMP="$DEPS/tmp"

BUILD_DIR="$DEPS"
DEPS_DIR="$TMP"

mkdir deps

if [[ -d "$TMP" ]]; then
    rm -rf "$TMP"
fi

mkdir $TMP
cd $TMP

###############################################################################
# emp-tool
###############################################################################
git clone "https://github.com/emp-toolkit/emp-tool.git" emp-tool
cd emp-tool
git checkout 802b5d4
sed -i '4i #include <cstdint>' emp-tool/utils/block.h
cmake . -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --target install -j
cd ..

###############################################################################
# emp-ot
###############################################################################
git clone https://github.com/emp-toolkit/emp-ot.git emp-ot
cd emp-ot
git checkout a603ca0
cmake $TMP/emp-ot -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_PREFIX_PATH=$DEPS \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build . --target install -j
cd ..

###############################################################################
# SEAL
###############################################################################
git clone https://github.com/microsoft/SEAL.git $DEPS_DIR/SEAL
cd $DEPS_DIR/SEAL
git switch --detach v4.1.2
patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/SEAL.patch -d $DEPS_DIR/SEAL/
cmake . -B build -DCMAKE_INSTALL_PREFIX=$BUILD_DIR \
    -DCMAKE_PREFIX_PATH=$BUILD_DIR -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF \
    -DSEAL_USE_ZSTD=ON -DCMAKE_BUILD_TYPE=Release -DSEAL_USE_INTEL_HEXL=ON \
    -DSEAL_BUILD_DEPS=ON -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target install --parallel 8

###############################################################################
# troy-nova
###############################################################################
if [[ "$1" = "-gpu" ]]; then
    git clone "https://github.com/lightbulb128/troy-nova.git" $DEPS_DIR/troy-nova
    cd $DEPS_DIR/troy-nova
    # c1913dd: conv2d with batched multiply-accumulate (one kernel per layer instead of one per ciphertext)
    git checkout c1913dd
    git submodule update --init extern/zstd
    patch --quiet --no-backup-if-mismatch -N -p1 -i $WORK_DIR/patch/troy-nova.patch -d $DEPS_DIR/troy-nova

    # bundled zstd, built position-independent: a system libzstd.a cannot go into the shared libtroy
    cmake -B build . -DCMAKE_INSTALL_PREFIX=$BUILD_DIR \
        -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" -DCMAKE_BUILD_TYPE=$BUILD_MODE \
        -DCMAKE_PREFIX_PATH=$BUILD_DIR -DTROY_PYBIND=OFF -DTROY_TEST=OFF \
        -DTROY_BENCH=OFF -DTROY_EXAMPLES=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_zstd=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    cmake --build build -t install -j
fi


rm -rf "$TMP"
