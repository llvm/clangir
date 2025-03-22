#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

__device__ int k;

__global__ void fn() {
  int i = 0;
  __shared__ int j;
  j = i;
  k = i;
}

// CIR: cir.global "private" internal dsolocal addrspace(offload_local) @_ZZ2fnvE1j : !s32i
// CIR: cir.func @_Z2fnv() {{.*}} {
// CIR:   %[[#Local:]] = cir.alloca !s32i, !cir.ptr<!s32i>
// CIR:   %[[#Shared:]] = cir.get_global @_ZZ2fnvE1j : !cir.ptr<!s32i, addrspace(offload_local)>
// CIR:   %[[#Converted:]] = cir.cast(address_space, %[[#Shared]] :
// CIR-SAME: !cir.ptr<!s32i, addrspace(offload_local)>), !cir.ptr<!s32i>
// CIR:   %[[#Tmp:]] = cir.load %[[#Local]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[#Tmp]], %[[#Converted]] : !s32i, !cir.ptr<!s32i>
// CIR: }

// LLVM: @_ZZ2fnvE1j = internal addrspace(3) global i32 undef
// LLVM: define dso_local ptx_kernel void @_Z2fnv() #{{.*}} {
// LLVM:   %[[#T1:]] = alloca i32, i64 1
// LLVM:   store i32 0, ptr %[[#T1]]
// LLVM:   %[[#T2:]] = load i32, ptr %[[#T1]]
// LLVM:   store i32 %[[#T2]], ptr addrspacecast (ptr addrspace(3) @_ZZ2fnvE1j to ptr)
// LLVM:   ret void
// LLVM: }
