#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^hipblas.*" \
  --allowlist-function="^hipblas.*" \
  --default-enum-style=rust \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/rocm/include \
  > sys.rs