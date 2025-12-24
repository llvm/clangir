// RUN: %clang_cc1 %s -fclangir -triple amdgcn-amd-amdhsa -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test kernel function with amdgpu_kernel calling convention
// CHECK: cir.func{{.*}} @simple_kernel() cc(amdgpu_kernel)
__kernel void simple_kernel() {}

// Test kernel with simple integer argument
// CHECK: cir.func{{.*}} @kernel_with_int(%arg{{[0-9]+}}: !s32i{{.*}}) cc(amdgpu_kernel)
__kernel void kernel_with_int(int x) {}

// Test kernel with pointer argument (should be in global address space)
// CHECK: cir.func{{.*}} @kernel_with_ptr(%arg{{[0-9]+}}: !cir.ptr<!s32i, lang_address_space(offload_global)>{{.*}}) cc(amdgpu_kernel)
__kernel void kernel_with_ptr(global int *ptr) {}

// Test device function (should NOT have amdgpu_kernel calling convention)
// CHECK: cir.func{{.*}} @device_fn
// CHECK-NOT: cc(amdgpu_kernel)
void device_fn(int x) {}
