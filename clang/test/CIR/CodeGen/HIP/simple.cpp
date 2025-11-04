#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -fhip-new-launch-api \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:              -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -emit-llvm -fhip-new-launch-api \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:              -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// Attribute for global_fn
// CIR-HOST: [[Kernel:#[a-zA-Z_0-9]+]] = {{.*}}#cir.cu.kernel_name<_Z9global_fni>{{.*}}


__host__ void host_fn(int *a, int *b, int *c) {}
// CIR-HOST: cir.func dso_local @_Z7host_fnPiS_S_
// CIR-DEVICE-NOT: cir.func dso_local @_Z7host_fnPiS_S_

__device__ void device_fn(int* a, double b, float c) {}
// CIR-HOST-NOT: cir.func dso_local @_Z9device_fnPidf
// CIR-DEVICE: cir.func dso_local @_Z9device_fnPidf

__global__ void global_fn(int a) {}
// CIR-DEVICE: @_Z9global_fni
// LLVM-DEVICE: define dso_local void @_Z9global_fni

// CIR-HOST: @_Z24__device_stub__global_fni{{.*}}extra([[Kernel]])
// CIR-HOST: %[[#CIRKernelArgs:]] = cir.alloca {{.*}}"kernel_args"
// CIR-HOST: %[[#Decayed:]] = cir.cast array_to_ptrdecay %[[#CIRKernelArgs]]
// CIR-HOST: cir.call @__hipPopCallConfiguration
// CIR-HOST: cir.get_global @_Z9global_fni : !cir.ptr<!cir.ptr<!cir.func<(!s32i)>>>
// CIR-HOST: cir.call @hipLaunchKernel

// LLVM-HOST: void @_Z24__device_stub__global_fni
// LLVM-HOST: %[[#KernelArgs:]] = alloca [1 x ptr], i64 1, align 16
// LLVM-HOST: %[[#GEP1:]] = getelementptr ptr, ptr %[[#KernelArgs]], i32 0
// LLVM-HOST: %[[#GEP2:]] = getelementptr [1 x ptr], ptr %[[#KernelArgs]], i32 0, i64 0
// LLVM-HOST: call i32 @__hipPopCallConfiguration
// LLVM-HOST: call i32 @hipLaunchKernel(ptr @_Z9global_fni 

int main() {
  global_fn<<<1, 1>>>(1);
}
// CIR-DEVICE-NOT: cir.func dso_local @main()

// CIR-HOST: cir.func dso_local @main()
// CIR-HOST: cir.call @_ZN4dim3C1Ejjj
// CIR-HOST: cir.call @_ZN4dim3C1Ejjj
// CIR-HOST: [[Push:%[0-9]+]] = cir.call @__hipPushCallConfiguration
// CIR-HOST: [[ConfigOK:%[0-9]+]] = cir.cast int_to_bool [[Push]]
// CIR-HOST: cir.if [[ConfigOK]] {
// CIR-HOST: } else {
// CIR-HOST:   [[Arg:%[0-9]+]] = cir.const #cir.int<1>
// CIR-HOST:   cir.call @_Z24__device_stub__global_fni([[Arg]])
// CIR-HOST: }

// LLVM-HOST: define dso_local i32 @main
// LLVM-HOST: alloca %struct.dim3
// LLVM-HOST: alloca %struct.dim3
// LLVM-HOST: call void @_ZN4dim3C1Ejjj
// LLVM-HOST: call void @_ZN4dim3C1Ejjj
// LLVM-HOST: %[[#ConfigOK:]] = call i32 @__hipPushCallConfiguration
// LLVM-HOST: %[[#ConfigCond:]] = icmp ne i32 %[[#ConfigOK]], 0
// LLVM-HOST: br i1 %[[#ConfigCond]], label %[[#Good:]], label %[[#Bad:]]
// LLVM-HOST: [[#Good]]:
// LLVM-HOST:   br label %[[#End:]]
// LLVM-HOST: [[#Bad]]:
// LLVM-HOST:   call void @_Z24__device_stub__global_fni(i32 1)
// LLVM-HOST:   br label %[[#End:]]
// LLVM-HOST: [[#End]]:
// LLVM-HOST:   %[[#]] = load i32
// LLVM-HOST:   ret i32

