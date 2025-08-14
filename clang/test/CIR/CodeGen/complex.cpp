// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

void complex_functional_cast() {
  using IntComplex = int _Complex;
  int _Complex a = IntComplex{};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.complex<#cir.int<0> : !s32i, #cir.int<0> : !s32i> : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[INIT]], align 4

void complex_comma_operator(int _Complex a, int _Complex b) {
  int _Complex c = (a, b);
}

// CIR: %[[COMPLEX_A:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX_B:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[TMP_B:.*]] = cir.load{{.*}} %[[COMPLEX_B]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP_B]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[COMPLEX_B]], align 4
// LLVM: store { i32, i32 } %[[TMP_B]], ptr %[[RESULT]], align 4
