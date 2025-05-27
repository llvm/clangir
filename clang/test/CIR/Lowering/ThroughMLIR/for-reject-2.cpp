// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void reject() {
  for (int i = 0; i < 100; i++, i++);
  // CHECK: %[[ALLOCA:.+]] = memref.alloca
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: memref.store %[[ZERO]], %[[ALLOCA]]
  // CHECK: %[[HUNDRED:.+]] = arith.constant 100
  // CHECK: scf.while : () -> () {
  // CHECK:   %[[TMP:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[TMP2:.+]] = arith.cmpi slt, %[[TMP]], %[[HUNDRED]]
  // CHECK:   scf.condition(%[[TMP2]])
  // CHECK: } do {
  // CHECK:   %[[TMP3:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[ONE:.+]] = arith.constant 1
  // CHECK:   %[[ADD:.+]] = arith.addi %[[TMP3]], %[[ONE]]
  // CHECK:   memref.store %[[ADD]], %[[ALLOCA]]
  // CHECK:   %[[LOAD:.+]] = memref.load %[[ALLOCA]]
  // CHECK:   %[[ONE2:.+]] = arith.constant 1
  // CHECK:   %[[ADD2:.+]] = arith.addi %[[LOAD]], %[[ONE2]]
  // CHECK:   memref.store %[[ADD2]], %[[ALLOCA]]
  // CHECK:   scf.yield
  // CHECK: }
}
