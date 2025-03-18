#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

__device__ int a;
// CIR-DEVICE: cir.global external addrspace(offload_global) @a = #cir.int<0>
// LLVM-DEVICE: @a = addrspace(1) externally_initialized global i32 0, align 4
// CIR-HOST: {{.*}}cir.global external @a = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<a>}{{.*}}

__shared__ int shared;
// CIR-DEVICE: cir.global external addrspace(offload_local) @shared = #cir.undef
// LLVM-DEVICE: @shared = addrspace(3) global i32 undef, align 4

__constant__ int b;
// CIR-DEVICE: cir.global constant external addrspace(offload_constant) @b = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// LLVM-DEVICE: @b = addrspace(4) externally_initialized constant i32 0, align 4
// CIR-HOST: {{.*}}cir.global external @b = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<b>}{{.*}}
