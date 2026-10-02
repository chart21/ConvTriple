#!/bin/bash

BUILD_TYPE=Release
# TRIPLE_BITLEN=64 ./build_cpu.sh: the 64-bit triples (hpmpc BITLENGTH=64 links build64)
BITLEN=${TRIPLE_BITLEN:-32}
BUILD_DIR=$([ "$BITLEN" = 64 ] && echo build64 || echo build)
FERRET_DIR="data"

if [[ ! -d $BUILD_DIR ]]; then
    cmake . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DTRIPLE_VERIFY=OFF \
        -DTRIPLE_COLOR=OFF -DUSE_APPROX_RESHARE=OFF -DTRIPLE_ZERO=ON \
        -DTRIPLE_GPU=OFF -DTRIPLE_BITLEN=$BITLEN \
        -DCMAKE_CXX_COMPILER=g++
fi

if [[ ! -d $FERRET_DIR ]]; then
    mkdir $FERRET_DIR
else
    rm -f $FERRET_DIR/*
fi

cmake --build $BUILD_DIR -j
