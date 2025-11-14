#!/usr/bin/env bash
set -e

# Build a shared library from fastcv_bridge.c
gcc -shared -fPIC \
  fastcv_bridge.c \
  -o libfastcv_bridge.so
