#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:              -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s


__shared__ int a;

// LLVM-DEVICE: @a = addrspace(3) {{.*}}

__device__ int b;

// LLVM-DEVICE: @b = addrspace(1) {{.*}}

__constant__ int c;

// LLVM-DEVICE: @c = addrspace(4) {{.*}}
