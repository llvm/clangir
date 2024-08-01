// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

global int a = 13;
// CIR-DAG: cir.global external addrspace(offload_global) @a = #cir.int<13> : !s32i
// LLVM-DAG: @a = addrspace(1) global i32 13

global int b = 15;
// CIR-DAG: cir.global external addrspace(offload_global) @b = #cir.int<15> : !s32i
// LLVM-DAG: @b = addrspace(1) global i32 15

kernel void test_get_global() {
  a = b;
  // CIR:      %[[#ADDRB:]] = cir.get_global @b : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-NEXT: %[[#LOADB:]] = cir.load %[[#ADDRB]] : !cir.ptr<!s32i, addrspace(offload_global)>, !s32i
  // CIR-NEXT: %[[#ADDRA:]] = cir.get_global @a : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-NEXT: cir.store %[[#LOADB]], %[[#ADDRA]] : !s32i, !cir.ptr<!s32i, addrspace(offload_global)>

  // LLVM:      %[[#LOADB:]] = load i32, ptr addrspace(1) @b, align 4
  // LLVM-NEXT: store i32 %[[#LOADB]], ptr addrspace(1) @a, align 4
}

kernel void test_static(int i) {
  static global int b = 15;
  // CIR-DAG: cir.global "private" internal dsolocal addrspace(offload_global) @func.b = #cir.int<15> : !s32i {alignment = 4 : i64}
  // LLVM-DAG: @func.b = internal addrspace(1) global i32 15

  local int c;
  // CIR-DAG: cir.global "private" internal dsolocal addrspace(offload_local) @func.c : !s32i {alignment = 4 : i64}
  // LLVM-DAG: @func.c = internal addrspace(3) global i32 undef

  // CIR-DAG: %[[#ADDRA:]] = cir.get_global @a : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-DAG: %[[#ADDRB:]] = cir.get_global @func.b : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-DAG: %[[#ADDRC:]] = cir.get_global @func.c : !cir.ptr<!s32i, addrspace(offload_local)>

  c = a;
  // CIR:      %[[#LOADA:]] = cir.load %[[#ADDRA]] : !cir.ptr<!s32i, addrspace(offload_global)>, !s32i
  // CIR-NEXT: cir.store %[[#LOADA]], %[[#ADDRC]] : !s32i, !cir.ptr<!s32i, addrspace(offload_local)>

  // LLVM:     %[[#LOADA:]] = load i32, ptr addrspace(1) @a, align 4
  // LLVM-NEXT: store i32 %[[#LOADA]], ptr addrspace(3) @func.c, align 4

  a = b;
  // CIR: %[[#LOADB:]] = cir.load %[[#ADDRB]] : !cir.ptr<!s32i, addrspace(offload_global)>, !s32i
  // CIR-NEXT: %[[#ADDRA:]] = cir.get_global @a : !cir.ptr<!s32i, addrspace(offload_global)>
  // CIR-NEXT: cir.store %[[#LOADB]], %[[#ADDRA]] : !s32i, !cir.ptr<!s32i, addrspace(offload_global)>

  // LLVM:      %[[#LOADB:]] = load i32, ptr addrspace(1) @func.b, align 4
  // LLVM-NEXT: store i32 %[[#LOADB]], ptr addrspace(1) @a, align 4
}
