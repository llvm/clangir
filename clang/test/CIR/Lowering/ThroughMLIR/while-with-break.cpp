// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void while_break() {
  int i = 0;
  while (i < 100) {
    i++;
    break;
    i++;
  }
  // This should be compiled into the condition `i < 100` and a single `i++`,
  // without the while-loop.

  // CHECK: memref.alloca_scope  {
  // CHECK:   %[[IV:.+]] = memref.load %alloca[]
  // CHECK:   %[[HUNDRED:.+]] = arith.constant 100
  // CHECK:   %[[_:.+]] = arith.cmpi slt, %[[IV]], %[[HUNDRED]]
  // CHECK:   memref.alloca_scope  {
  // CHECK:     %[[IV2:.+]] = memref.load %alloca[]
  // CHECK:     %[[ONE:.+]] = arith.constant 1
  // CHECK:     %[[INCR:.+]] = arith.addi %[[IV2]], %[[ONE]]
  // CHECK:     memref.store %[[INCR]], %alloca[]
  // CHECK:   }
  // CHECK: }
}
