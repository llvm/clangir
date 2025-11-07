#include "cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I$(dirname %s)/../Inputs/ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip  \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I$(dirname %s)/../Inputs/ -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s


__shared__ int a;
// LLVM-DEVICE: @a = addrspace(3) global i32 undef, align 4
// OGCG-DEVICE: @a = addrspace(3) global i32 undef, align 4

__device__ int b;
// LLVM-DEVICE: @b = addrspace(1) externally_initialized global i32 0, align 4
// OGCG-DEVICE: @b = addrspace(1) externally_initialized global i32 0, align 4

__constant__ int c;
// LLVM-DEVICE: @c = addrspace(4) externally_initialized constant i32 0, align 4
// OGCG-DEVICE: @c = addrspace(4) externally_initialized constant i32 0, align 4

