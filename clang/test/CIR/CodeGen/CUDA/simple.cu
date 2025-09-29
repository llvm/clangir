#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu  \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

// Attribute for global_fn
// CIR-HOST: [[Kernel:#[a-zA-Z_0-9]+]] = {{.*}}#cir.cu.kernel_name<_Z9global_fni>{{.*}}

__host__ void host_fn(int *a, int *b, int *c) {}
// CIR-HOST: cir.func dso_local @_Z7host_fnPiS_S_
// CIR-DEVICE-NOT: cir.func dso_local @_Z7host_fnPiS_S_

__device__ void device_fn(int* a, double b, float c) {}
// CIR-HOST-NOT: cir.func dso_local @_Z9device_fnPidf
// CIR-DEVICE: cir.func dso_local @_Z9device_fnPidf

__global__ void global_fn(int a) {}
// CIR-DEVICE: @_Z9global_fni({{.*}} cc(ptx_kernel)
// LLVM-DEVICE: define dso_local ptx_kernel void @_Z9global_fni
// OGCG-DEVICE: define dso_local ptx_kernel void @_Z9global_fni

// Check for device stub emission.

// CIR-HOST: @_Z24__device_stub__global_fni{{.*}}extra([[Kernel]])
// CIR-HOST: %[[#CIRKernelArgs:]] = cir.alloca {{.*}}"kernel_args"
// CIR-HOST: %[[#Decayed:]] = cir.cast(array_to_ptrdecay, %[[#CIRKernelArgs]]
// CIR-HOST: cir.call @__cudaPopCallConfiguration
// CIR-HOST: cir.get_global @_Z24__device_stub__global_fni
// CIR-HOST: cir.call @cudaLaunchKernel

// LLVM-HOST: void @_Z24__device_stub__global_fni
// LLVM-HOST: %[[#KernelArgs:]] = alloca [1 x ptr], i64 1, align 16
// LLVM-HOST: %[[#GEP1:]] = getelementptr ptr, ptr %[[#KernelArgs]], i32 0
// LLVM-HOST: %[[#GEP2:]] = getelementptr [1 x ptr], ptr %[[#KernelArgs]], i32 0, i64 0
// LLVM-HOST: call i32 @__cudaPopCallConfiguration
// LLVM-HOST: call i32 @cudaLaunchKernel(ptr @_Z24__device_stub__global_fni

// OGCG-HOST: void @_Z24__device_stub__global_fni
// OGCG-HOST: %kernel_args = alloca ptr, i64 1, align 16
// OGCG-HOST: getelementptr ptr, ptr %kernel_args, i32 0
// OGCG-HOST: call i32 @__cudaPopCallConfiguration
// OGCG-HOST: call noundef i32 @cudaLaunchKernel(ptr noundef @_Z24__device_stub__global_fni


int main() {
  global_fn<<<1, 1>>>(1);
}
// CIR-DEVICE-NOT: cir.func dso_local @main()

// CIR-HOST: cir.func dso_local @main()
// CIR-HOST: cir.call @_ZN4dim3C1Ejjj
// CIR-HOST: cir.call @_ZN4dim3C1Ejjj
// CIR-HOST: [[Push:%[0-9]+]] = cir.call @__cudaPushCallConfiguration
// CIR-HOST: [[ConfigOK:%[0-9]+]] = cir.cast(int_to_bool, [[Push]]
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
// LLVM-HOST: %[[#ConfigOK:]] = call i32 @__cudaPushCallConfiguration
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

// OGCG-HOST: define dso_local noundef i32 @main
// OGCG-HOST: alloca %struct.dim3, align 4
// OGCG-HOST: alloca %struct.dim3, align 4
// OGCG-HOST: call void @_ZN4dim3C1Ejjj
// OGCG-HOST: call void @_ZN4dim3C1Ejjj
// OGCG-HOST: %call = call i32 @__cudaPushCallConfiguration
// OGCG-HOST: %tobool = icmp ne i32 %call, 0
// OGCG-HOST: br i1 %tobool, label %kcall.end, label %kcall.configok
// OGCG-HOST: kcall.configok:
// OGCG-HOST:   call void @_Z24__device_stub__global_fni(i32 noundef 1)
// OGCG-HOST:   br label %kcall.end
// OGCG-HOST: kcall.end:
// OGCG-HOST:   %{{[0-9]+}} = load i32, ptr %retval, align 4
// OGCG-HOST:   ret i32