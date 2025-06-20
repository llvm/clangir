// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
 
void while_continue() {
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

void while_continue_2() {
  int i = 0;
  while (i < 10) {
    if (i == 5) {
      i += 3;
      continue;
    }
  
    i++;
  }
  // The final i++ will have a `if (!(i == 5))` guarded against it.

  // CHECK: do {
  // CHECK:   %[[NOTALLOCA:.+]] = memref.alloca
  // CHECK:   memref.alloca_scope  {
  // CHECK:     memref.alloca_scope  {
  // CHECK:       %[[IV:.+]] = memref.load %[[IVADDR:.+]][]
  // CHECK:       %[[FIVE:.+]] = arith.constant 5
  // CHECK:       %[[COND:.+]] = arith.cmpi eq, %[[IV]], %[[FIVE]]
  // CHECK:       %true = arith.constant true
  // CHECK:       %[[NOT:.+]] = arith.xori %true, %[[COND]]
  // CHECK:       %[[EXT:.+]] = arith.extui %[[NOT]] : i1 to i8
  // CHECK:       memref.store %[[EXT]], %[[NOTALLOCA]]
  // CHECK:       scf.if %[[COND]] {
  // CHECK:         %[[THREE:.+]] = arith.constant 3
  // CHECK:         %[[IV2:.+]] = memref.load %[[IVADDR]]
  // CHECK:         %[[TMP:.+]] = arith.addi %[[IV2]], %[[THREE]]
  // CHECK:         memref.store %[[TMP]], %[[IVADDR]]
  // CHECK:       }
  // CHECK:     }
  // CHECK:     %[[NOTCOND:.+]] = memref.load %[[NOTALLOCA]]
  // CHECK:     %[[TRUNC:.+]] = arith.trunci %[[NOTCOND]] : i8 to i1
  // CHECK:     scf.if %[[TRUNC]] {
  // CHECK:       %[[IV3:.+]] = memref.load %[[IVADDR]]
  // CHECK:       %[[ONE:.+]] = arith.constant 1
  // CHECK:       %[[TMP2:.+]] = arith.addi %[[IV3]], %[[ONE]]
  // CHECK:       memref.store %[[TMP2]], %[[IVADDR]]
  // CHECK:     }
  // CHECK:   }
  // CHECK:   scf.yield
  // CHECK: }
}
