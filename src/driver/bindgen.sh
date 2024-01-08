#!/bin/bash
set -exu

bindgen \
  --allowlist-type="^hip.*" \
  --allowlist-function="^hip.*" \
  --allowlist-var="^hip.*" \
  --default-enum-style=rust \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --use-core \
  wrapper.h -- -I/opt/rocm/include \
  > sys.rs

#   --dynamic-loading rocm \
#  --dynamic-link-require-all \
#  --prefix-link-name hip \

#  --allowlist-type="^HIP.*" \
#  --allowlist-type="^cuuint(32|64)_t" \
#  --allowlist-type="^hipError_enum" \
#  --allowlist-type="^hip.*Complex$" \
#  --allowlist-type="^hip.*" \
#  --allowlist-type="^libraryPropertyType.*" \
#  b
#  --allowlist-function="^hip.*" \

#bindgen \
#  --default-enum-style=rust \
#  --no-doc-comments \
#  --with-derive-default \
#  --with-derive-eq \
#  --with-derive-hash \
#  --with-derive-ord \
#  --use-core \
#  wrapper.h -- -I/opt/rocm/include \
#  > sys.rs

## The rocm include path may need to be adjusted
#bindgen wrapper.h \
#        --raw-line "#![allow(non_camel_case_types)]" \
#        --raw-line "#![allow(non_upper_case_globals)]" \
#        --raw-line "#![allow(non_snake_case)]" \
#        --rustified-enum "hip.*" \
#        --generate-block \
#        --ctypes-prefix "::libc" \
#        --with-derive-default \
#        --with-derive-eq \
#        --with-derive-ord \
#        --with-derive-hash \
#        -o sys.rs \
#        -- -I /opt/rocm/include/
