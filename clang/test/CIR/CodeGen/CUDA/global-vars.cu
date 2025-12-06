#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

// HIP tests
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            -x hip %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE-HIP --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -x hip %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE-HIP --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST-HIP --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST-HIP --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x hip -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST-HIP --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            -x hip %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE-HIP --input-file=%t.ll %s

__shared__ int shared;
// CIR-DEVICE: cir.global external{{.*}}lang_address_space(offload_local) @shared = #cir.undef
// LLVM-DEVICE: @shared = addrspace(3) global i32 undef, align 4
// CIR-HOST: cir.global{{.*}}@shared = #cir.undef : !s32i {alignment = 4 : i64}
// CIR-HOST-NOT: cu.shadow_name
// LLVM-HOST: @shared = internal global i32 undef, align 4
// OGCG-HOST: @shared = internal global i32
// OGCG-DEVICE: @shared = addrspace(3) global i32 undef, align 4
// CIR-DEVICE-HIP: cir.global external{{.*}}lang_address_space(offload_local) @shared = #cir.undef
// LLVM-DEVICE-HIP: @shared = addrspace(3) global i32 undef, align 4
// CIR-HOST-HIP: cir.global{{.*}}@shared = #cir.undef : !s32i {alignment = 4 : i64}
// CIR-HOST-HIP-NOT: cu.shadow_name
// LLVM-HOST-HIP: @shared = internal global i32 undef, align 4
// OGCG-HOST-HIP: @shared = internal global i32
// OGCG-DEVICE-HIP: @shared = addrspace(3) global i32 undef, align 4

__constant__ int b;
// CIR-DEVICE: cir.global constant external{{.*}}lang_address_space(offload_constant) @b = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// LLVM-DEVICE: @b = addrspace(4) externally_initialized constant i32 0, align 4
// CIR-HOST: cir.global{{.*}}"private"{{.*}}internal{{.*}}@b = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<b>}
// LLVM-HOST: @b = internal global i32 undef, align 4
// OGCG-HOST: @b = internal global i32
// OGCG-DEVICE: @b = addrspace(4) externally_initialized constant i32 0, align 4
// CIR-DEVICE-HIP: cir.global constant external{{.*}}lang_address_space(offload_constant) @b = #cir.int<0> : !s32i {alignment = 4 : i64, cu.externally_initialized = #cir.cu.externally_initialized}
// LLVM-DEVICE-HIP: @b = addrspace(4) externally_initialized constant i32 0, align 4
// CIR-HOST-HIP: cir.global{{.*}}"private"{{.*}}internal{{.*}}@b = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<b>}
// LLVM-HOST-HIP: @b = internal global i32 undef, align 4
// OGCG-HOST-HIP: @b = internal global i32
// OGCG-DEVICE-HIP: @b = addrspace(4) externally_initialized constant i32 0, align 4

// External device variables should remain external on host side (they're just declarations)
// Note: External declarations may not appear in output if they're not referenced
extern __device__ int ext_device_var;
// CIR-HOST-NOT: cir.global{{.*}}@ext_device_var
// LLVM-HOST-NOT: @ext_device_var
// OGCG-HOST-NOT: @ext_device_var
// OGCG-DEVICE-NOT: @ext_device_var
// CIR-HOST-HIP-NOT: cir.global{{.*}}@ext_device_var
// LLVM-HOST-HIP-NOT: @ext_device_var
// OGCG-HOST-HIP-NOT: @ext_device_var
// OGCG-DEVICE-HIP-NOT: @ext_device_var

extern __constant__ int ext_constant_var;
// CIR-HOST-NOT: cir.global{{.*}}@ext_constant_var
// LLVM-HOST-NOT: @ext_constant_var
// OGCG-HOST-NOT: @ext_constant_var
// OGCG-DEVICE-NOT: @ext_constant_var
// CIR-HOST-HIP-NOT: cir.global{{.*}}@ext_constant_var
// LLVM-HOST-HIP-NOT: @ext_constant_var
// OGCG-HOST-HIP-NOT: @ext_constant_var
// OGCG-DEVICE-HIP-NOT: @ext_constant_var

// External device variables with definitions should be internal on host
extern __device__ int ext_device_var_def;
__device__ int ext_device_var_def = 1;
// CIR-DEVICE: cir.global external{{.*}}lang_address_space(offload_global) @ext_device_var_def = #cir.int<1>
// LLVM-DEVICE: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4
// CIR-HOST: cir.global{{.*}}"private"{{.*}}internal{{.*}}@ext_device_var_def = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<ext_device_var_def>}
// LLVM-HOST: @ext_device_var_def = internal global i32 undef, align 4
// OGCG-HOST: @ext_device_var_def = internal global i32
// OGCG-DEVICE: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4
// CIR-DEVICE-HIP: cir.global external{{.*}}lang_address_space(offload_global) @ext_device_var_def = #cir.int<1>
// LLVM-DEVICE-HIP: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4
// CIR-HOST-HIP: cir.global{{.*}}"private"{{.*}}internal{{.*}}@ext_device_var_def = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<ext_device_var_def>}
// LLVM-HOST-HIP: @ext_device_var_def = internal global i32 undef, align 4
// OGCG-HOST-HIP: @ext_device_var_def = internal global i32
// OGCG-DEVICE-HIP: @ext_device_var_def = addrspace(1) externally_initialized global i32 1, align 4

extern __constant__ int ext_constant_var_def;
__constant__ int ext_constant_var_def = 2;
// CIR-DEVICE: cir.global constant external{{.*}}lang_address_space(offload_constant) @ext_constant_var_def = #cir.int<2>
// LLVM-DEVICE: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4
// OGCG-DEVICE: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4
// CIR-HOST: cir.global{{.*}}"private"{{.*}}internal{{.*}}@ext_constant_var_def = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<ext_constant_var_def>}
// LLVM-HOST: @ext_constant_var_def = internal global i32 undef, align 4
// OGCG-HOST: @ext_constant_var_def = internal global i32
// CIR-DEVICE-HIP: cir.global constant external{{.*}}lang_address_space(offload_constant) @ext_constant_var_def = #cir.int<2>
// LLVM-DEVICE-HIP: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4
// CIR-HOST-HIP: cir.global{{.*}}"private"{{.*}}internal{{.*}}@ext_constant_var_def = #cir.undef : !s32i {alignment = 4 : i64, cu.shadow_name = #cir.cu.shadow_name<ext_constant_var_def>}
// LLVM-HOST-HIP: @ext_constant_var_def = internal global i32 undef, align 4
// OGCG-HOST-HIP: @ext_constant_var_def = internal global i32
// OGCG-DEVICE-HIP: @ext_constant_var_def = addrspace(4) externally_initialized constant i32 2, align 4

// Regular host variables should NOT be internalized
int host_var;
// CIR-HOST: cir.global external @host_var = #cir.int<0> : !s32i
// LLVM-HOST: @host_var = global i32 0, align 4
// OGCG-HOST: @host_var ={{.*}} global i32

// CIR-DEVICE-NOT: cir.global{{.*}}@host_var
// LLVM-DEVICE-NOT: @host_var
// OGCG-DEVICE-NOT: @host_var

// CIR-HOST-HIP: cir.global external @host_var = #cir.int<0> : !s32i
// LLVM-HOST-HIP: @host_var = global i32 0, align 4
// OGCG-HOST-HIP: @host_var ={{.*}} global i32

// CIR-DEVICE-HIP-NOT: cir.global{{.*}}@host_var
// LLVM-DEVICE-HIP-NOT: @host_var
// OGCG-DEVICE-HIP-NOT: @host_var

// External host variables should remain external (may not appear if not referenced)
extern int ext_host_var;
// CIR-HOST-NOT: cir.global{{.*}}@ext_host_var
// LLVM-HOST-NOT: @ext_host_var
// OGCG-HOST-NOT: @ext_host_var

// CIR-DEVICE-NOT: cir.global{{.*}}@ext_host_var
// LLVM-DEVICE-NOT: @ext_host_var
// OGCG-DEVICE-NOT: @ext_host_var

// CIR-HOST-HIP-NOT: cir.global{{.*}}@ext_host_var
// LLVM-HOST-HIP-NOT: @ext_host_var
// OGCG-HOST-HIP-NOT: @ext_host_var

// CIR-DEVICE-HIP-NOT: cir.global{{.*}}@ext_host_var
// LLVM-DEVICE-HIP-NOT: @ext_host_var
// OGCG-DEVICE-HIP-NOT: @ext_host_var
