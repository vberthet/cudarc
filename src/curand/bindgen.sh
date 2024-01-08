#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^hiprand.*" \
  --allowlist-var="^hiprand.*" \
  --allowlist-function="^hiprand.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/rocm/include -x c++ -std=c++14  \
  > sys.rs