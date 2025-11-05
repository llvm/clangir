#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:              -emit-cir %s -o %t.ll
// RUN: FileCheck --check-prefix=CIR --input-file=%t.ll %s

__global__ void fn() {
  int i = 0;
  __shared__ int j;
  j = i;
}

// CIR: cir.global "private" internal dso_local addrspace(offload_local) @_ZZ2fnvE1j : !s32i
// CIR: cir.func dso_local @_Z2fnv
// CIR: [[Local:%[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR: [[Shared:%[0-9]+]] = cir.get_global @_ZZ2fnvE1j : !cir.ptr<!s32i, addrspace(offload_local)>
// CIR: [[Tmp:%[0-9]+]] = cir.load {{.*}} [[Local]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store{{.*}} [[Tmp]], [[Shared]] : !s32i, !cir.ptr<!s32i, addrspace(offload_local)>
