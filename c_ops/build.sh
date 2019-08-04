#!/bin/bash

set -e

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cp libChinese_aster.* ..