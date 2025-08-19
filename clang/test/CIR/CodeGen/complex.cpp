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

void complex_cxx_scalar_value_init_expr() {
  using IntComplex = int _Complex;
  int _Complex a = IntComplex();
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.zero : !cir.complex<!s32i>
// CIR: cir.store align(4) %[[COMPLEX]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[INIT]], align 4

void complex_abstract_condition(bool cond, int _Complex a, int _Complex b) {
  int _Complex c = cond ? a : b;
}

// CIR: %[[COND:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cond", init]
// CIR: %[[COMPLEX_A:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["a", init]
// CIR: %[[COMPLEX_B:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["b", init]
// CIR: %[[RESULT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[TMP_COND:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR: %[[RESULT_VAL:.*]] = cir.ternary(%[[TMP_COND]], true {
// CIR:   %[[TMP_A:.*]] = cir.load{{.*}} %[[COMPLEX_A]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:   cir.yield %[[TMP_A]] : !cir.complex<!s32i>
// CIR: }, false {
// CIR:   %[[TMP_B:.*]] = cir.load{{.*}} %[[COMPLEX_B]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:   cir.yield %[[TMP_B]] : !cir.complex<!s32i>
// CIR: }) : (!cir.bool) -> !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RESULT_VAL]], %[[RESULT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[COND:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[COMPLEX_A:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPLEX_B:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[RESULT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_COND:.*]] = load i8, ptr %[[COND]], align 1
// LLVM: %[[COND_VAL:.*]] = trunc i8 %[[TMP_COND]] to i1
// LLVM: br i1 %[[COND_VAL]], label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:  %[[TMP_A:.*]] = load { i32, i32 }, ptr %[[COMPLEX_A]], align 4
// LLVM:  br label %[[END_BB:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:  %[[TMP_B:.*]] = load { i32, i32 }, ptr %[[COMPLEX_B]], align 4
// LLVM:  br label %[[END_BB]]
// LLVM: [[END_BB]]:
// LLVM: %[[RESULT_VAL:.*]] = phi { i32, i32 } [ %[[TMP_B]], %[[FALSE_BB]] ], [ %[[TMP_A]], %[[TRUE_BB]] ]
// LLVM: store { i32, i32 } %[[RESULT_VAL]], ptr %[[RESULT]], align 4

int _Complex complex_real_operator_on_rvalue() {
  int real = __real__ complex_real_operator_on_rvalue();
  return {};
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["__retval"]
// CIR: %[[REAL_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["real", init]
// CIR: %[[CALL:.*]] = cir.call @_Z31complex_real_operator_on_rvaluev() : () -> !cir.complex<!s32i>
// CIR: %[[REAL:.*]] = cir.complex.real %[[CALL]] : !cir.complex<!s32i> -> !s32i
// CIR: cir.store{{.*}} %[[REAL]], %[[REAL_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[RET_COMPLEX:.*]] = cir.const #cir.complex<#cir.int<0> : !s32i, #cir.int<0> : !s32i> : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RET_COMPLEX]], %[[RET_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.return %[[TMP_RET]] : !cir.complex<!s32i>

// LLVM: %[[RET_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[REAL_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[CALL:.*]] = call { i32, i32 } @_Z31complex_real_operator_on_rvaluev()
// LLVM: %[[REAL:.*]] = extractvalue { i32, i32 } %[[CALL]], 0
// LLVM: store i32 %[[REAL]], ptr %[[REAL_ADDR]], align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[RET_ADDR]], align 4
// LLVM: %[[TMP_RET:.*]] = load { i32, i32 }, ptr %[[RET_ADDR]], align 4
// LLVM: ret { i32, i32 } %[[TMP_RET]]

int _Complex complex_imag_operator_on_rvalue() {
  int imag = __imag__ complex_imag_operator_on_rvalue();
  return {};
}

// CIR: %[[RET_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["__retval"]
// CIR: %[[IMAG_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["imag", init]
// CIR: %[[CALL:.*]] = cir.call @_Z31complex_imag_operator_on_rvaluev() : () -> !cir.complex<!s32i>
// CIR: %[[IMAG:.*]] = cir.complex.imag %[[CALL]] : !cir.complex<!s32i> -> !s32i
// CIR: cir.store{{.*}} %[[IMAG]], %[[IMAG_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[RET_COMPLEX:.*]] = cir.const #cir.complex<#cir.int<0> : !s32i, #cir.int<0> : !s32i> : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[RET_COMPLEX]], %[[RET_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.return %[[TMP_RET]] : !cir.complex<!s32i>

// LLVM: %[[RET_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[IMAG_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[CALL:.*]] = call { i32, i32 } @_Z31complex_imag_operator_on_rvaluev()
// LLVM: %[[IMAG:.*]] = extractvalue { i32, i32 } %[[CALL]], 1
// LLVM: store i32 %[[IMAG]], ptr %[[IMAG_ADDR]], align 4
// LLVM: store { i32, i32 } zeroinitializer, ptr %[[RET_ADDR]], align 4
// LLVM: %[[TMP_RET:.*]] = load { i32, i32 }, ptr %[[RET_ADDR]], align 4
// LLVM: ret { i32, i32 } %[[TMP_RET]]
