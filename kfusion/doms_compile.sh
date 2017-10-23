#!/bin/bash
g++ src/cpp/preprocess.cpp src/cpp/render.cpp src/cpp/kernels.cpp -I include/ -I thirdparty/ -std=c++11 -lm -fdump-rtl-expand -o doms.sh 2> doms_debug.txt