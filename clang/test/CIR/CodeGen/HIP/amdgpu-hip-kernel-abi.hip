#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:            -I%S/../../Inputs/ -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test simple kernel
// CHECK: cir.func{{.*}} @_Z13simple_kerneli(%arg{{[0-9]+}}: !s32i{{.*}}) cc(amdgpu_kernel)
__global__ void simple_kernel(int x) {}

// Test kernel with pointer
// CHECK: cir.func{{.*}} @_Z15kernel_with_ptrPi(%arg{{[0-9]+}}: !cir.ptr<!s32i{{.*}}>{{.*}}) cc(amdgpu_kernel)
__global__ void kernel_with_ptr(int *ptr) {}

// Test kernel with multiple args
// CHECK: cir.func{{.*}} @_Z16kernel_multi_argifdPi(%arg{{[0-9]+}}: !s32i{{.*}}, %arg{{[0-9]+}}: !cir.float{{.*}}, %arg{{[0-9]+}}: !cir.double{{.*}}, %arg{{[0-9]+}}: !cir.ptr<!s32i{{.*}}>{{.*}}) cc(amdgpu_kernel)
__global__ void kernel_multi_arg(int a, float b, double c, int *d) {}

// Test device function (NOT a kernel)
// CHECK: cir.func{{.*}} @_Z9device_fni(%arg{{[0-9]+}}: !s32i{{.*}})
// CHECK-NOT: cc(amdgpu_kernel)
__device__ void device_fn(int x) {}

// Test device function with return value
// CHECK: cir.func{{.*}} @_Z13device_fn_reti(%arg{{[0-9]+}}: !s32i{{.*}}) -> !s32i
// CHECK-NOT: cc(amdgpu_kernel)
__device__ int device_fn_ret(int x) { return x + 1; }
