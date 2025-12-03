// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
//
//
// Static local initialization divergence:
// Missing noundef attribute on function declaration and call
//
// CodeGen:
//   declare noundef i32 @_Z3fnAv()
//   %call = call noundef i32 @_Z3fnAv()
//
// CIR:
//   declare i32 @_Z3fnAv()  (missing noundef)
//   %8 = call i32 @_Z3fnAv()  (missing noundef)

// CHECK-DAG: %{{.*}} = call noundef i32 @_Z3fnAv()
// CHECK-DAG: declare noundef i32 @_Z3fnAv()
int fnA();

void foo() {
  static int val = fnA();
}
