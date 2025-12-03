// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll | FlileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll | FileCheck %s
//
// XFAIL: *
//
// Static local initialization divergence:
// 1. Missing noundef attribute on function declaration and call
//
// CodeGen:
//   declare noundef i32 @_Z3fnAv()
//   %call = call noundef i32 @_Z3fnAv()
//
// CIR:
//   declare i32 @_Z3fnAv()  (missing noundef)
//   %8 = call i32 @_Z3fnAv()  (missing noundef)

// DIFF-DAG: declare noundef i32 @_Z3fnAv()
// DIFF-DAG  %call = call noundef i32 @_Z3fnAv()

int fnA();

void foo() {
  static int val = fnA();
}
