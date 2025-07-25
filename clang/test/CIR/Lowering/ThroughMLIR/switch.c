// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void fallthrough() {
  int i = 0;
  switch (i) {
  case 2:
    i++;
  case 3:
    i++;
    break;
  case 8:
    i++;
  }

  // This should copy the `i++; break` in case 3 to case 2.

  // CHECK: memref.alloca_scope  {
  // CHECK:   %[[I:.+]] = memref.load %alloca[]
  // CHECK:   %[[CASTED:.+]] = arith.index_cast %[[I]]
  // CHECK:   scf.index_switch %[[CASTED]]
  // CHECK:   case 2 {
  // CHECK:     %[[I:.+]] = memref.load %alloca[]
  // CHECK:     %[[ONE:.+]] = arith.constant 1
  // CHECK:     %[[ADD:.+]] = arith.addi %[[I]], %[[ONE]]
  // CHECK:     memref.store %[[ADD]], %alloca[]
  // CHECK:     %[[I:.+]] = memref.load %alloca[]
  // CHECK:     %[[ONE:.+]] = arith.constant 1
  // CHECK:     %[[ADD:.+]] = arith.addi %[[I]], %[[ONE]]
  // CHECK:     memref.store %[[ADD]], %alloca[]
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK:   case 3 {
  // CHECK:     %[[I:.+]] = memref.load %alloca[]
  // CHECK:     %[[ONE:.+]] = arith.constant 1
  // CHECK:     %[[ADD:.+]] = arith.addi %[[I]], %[[ONE]]
  // CHECK:     memref.store %[[ADD]], %alloca[]
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK:   case 8 {
  // CHECK:     %[[I:.+]] = memref.load %alloca[]
  // CHECK:     %[[ONE:.+]] = arith.constant 1
  // CHECK:     %[[ADD:.+]] = arith.addi %[[I]], %[[ONE]]
  // CHECK:     memref.store %[[ADD]], %alloca[]
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK:   default {
  // CHECK:   }
  // CHECK: }
}
