// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// CIR: @func
// LLVM: @func
kernel void func(global int *data) {
    local int arr[32];
    local int *ptr = arr;
    // CIR:      cir.cast(array_to_ptrdecay, %{{[0-9]+}} : !cir.ptr<!cir.array<!s32i x 32>, addrspace(offload_local)>), !cir.ptr<!s32i, addrspace(offload_local)>
    // CIR-NEXT: cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.ptr<!s32i, addrspace(offload_local)>, !cir.ptr<!cir.ptr<!s32i, addrspace(offload_local)>, addrspace(offload_private)>
    // LLVM: store ptr addrspace(3) @func.arr, ptr %{{[0-9]+}}
}
