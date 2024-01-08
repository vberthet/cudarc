#!/bin/bash
# Requires rust-bindgen 0.68.1 or superior
set -exu
BINDGEN_EXTRA_CLANG_ARGS="-D__CUDA_BF16_TYPES_EXIST__" \
bindgen \
  --allowlist-type="^hipblasLt.*" \
  --allowlist-var="^hipblasLt.*" \
  --allowlist-function="^hipblasLt.*" \
  --default-enum-style=rust \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/rocm/include -x c++ -std=c++14  \
  > sys.rs
