#!/bin/bash

if [[ -d build ]]; then
    cd build
    ctest
    cd ..
fi

