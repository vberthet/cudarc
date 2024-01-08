#!/bin/bash
set -exu
BINDGEN_EXTRA_CLANG_ARGS="-D__CUDA_BF16_TYPES_EXIST__" \
bindgen \
  --allowlist-type="^nccl.*" \
  --allowlist-var="^nccl.*" \
  --allowlist-function="^nccl.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/rocm/include \
  > sys.rs
