// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.orig.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.orig.ll %s
//
// Test for cast expressions as l-values (const_cast, reinterpret_cast, etc.)

void const_cast_lvalue() {
  const int x = 0;
  const_cast<int&>(x) = 1;
}

// CIR-LABEL: cir.func dso_local @_Z17const_cast_lvaluev
// CIR:   %[[X:.*]] = cir.alloca !s32i, {{.*}}, ["x", init, const]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store {{.*}} %[[ZERO]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store {{.*}} %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return

// LLVM-LABEL: define {{.*}}void @_Z17const_cast_lvaluev
// LLVM:   %[[X:.*]] = alloca i32
// LLVM:   store i32 0, ptr %[[X]]
// LLVM:   store i32 1, ptr %[[X]]
// LLVM:   ret void

void reinterpret_cast_lvalue() {
  long x = 0;
  reinterpret_cast<int&>(x) = 1;
}

// CIR-LABEL: cir.func dso_local @_Z23reinterpret_cast_lvaluev
// CIR:   %[[X:.*]] = cir.alloca !s64i, {{.*}}, ["x", init]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store {{.*}} %{{.*}}, %[[X]] : !s64i, !cir.ptr<!s64i>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   %[[CAST:.*]] = cir.cast bitcast %[[X]] : !cir.ptr<!s64i> -> !cir.ptr<!s32i>
// CIR:   cir.store {{.*}} %[[ONE]], %[[CAST]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return

// LLVM-LABEL: define {{.*}}void @_Z23reinterpret_cast_lvaluev
// LLVM:   %[[X:.*]] = alloca i64
// LLVM:   store i64 0, ptr %[[X]]
// LLVM:   store i32 1, ptr %[[X]]
// LLVM:   ret void
