// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void f() {}

void reject() {
  for (int i = 0; i < 100; i++, f());
  // CHECK: %[[ALLOCA:.+]] = memref.alloca
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: memref.store %[[ZERO]], %[[ALLOCA]]
  // CHECK: %[[HUNDRED:.+]] = arith.constant 100
  // CHECK: scf.while : () -> () {
  // CHECK:   %[[TMP:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[TMP1:.+]] = arith.cmpi slt, %0, %[[HUNDRED]]
  // CHECK:   scf.condition(%[[TMP1]])
  // CHECK: } do {
  // CHECK:   %[[TMP2:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[ONE:.+]] = arith.constant 1
  // CHECK:   %[[TMP3:.+]] = arith.addi %[[TMP2]], %[[ONE]]
  // CHECK:   memref.store %[[TMP3]], %[[ALLOCA]]
  // CHECK:   func.call @_Z1fv()
  // CHECK:   scf.yield
  // CHECK: }
}
