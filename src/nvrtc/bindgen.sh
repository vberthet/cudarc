#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^hiprtc.*" \
  --allowlist-function="^hiprtc.*" \
  --default-enum-style=rust \
  --no-doc-comments \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/rocm/include \
  > sys.rs