// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
 
void for_with_break() {
  int i = 0;
  while (i < 100) {
    i++;
    continue;
    i++;
  }
  // Only the first `i++` will be emitted.

  // CHECK: scf.while : () -> () {
  // CHECK:   %[[TMP0:.+]] = memref.load %alloca[]
  // CHECK:   %[[HUNDRED:.+]] = arith.constant 100
  // CHECK:   %[[TMP1:.+]] = arith.cmpi slt, %[[TMP0]], %[[HUNDRED]]
  // CHECK:   scf.condition(%[[TMP1]])
  // CHECK: } do {
  // CHECK:   memref.alloca_scope  {
  // CHECK:     %[[TMP2:.+]] = memref.load %alloca[]
  // CHECK:     %[[ONE:.+]] = arith.constant 1
  // CHECK:     %[[TMP3:.+]] = arith.addi %[[TMP2]], %[[ONE]]
  // CHECK:     memref.store %[[TMP3]], %alloca[]
  // CHECK:   }
  // CHECK:   scf.yield
  // CHECK: }
}
