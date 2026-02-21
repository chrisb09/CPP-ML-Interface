#!/bin/sh
# configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release

# build (generator-agnostic; passes -j to underlying tool)
cmake --build build -- -j"$(nproc)"
cmake --build build_release -- -j"$(nproc)"