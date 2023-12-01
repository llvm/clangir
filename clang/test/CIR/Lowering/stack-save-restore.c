// RUN: cir-opt %s -cir-to-llvm -o - | FileCheck %s -check-prefix=MLIR


!u8i = !cir.int<u, 8>

module  {
  cir.func @foo() {
    %0 = cir.alloca !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>, ["saved_stack"] {alignment = 8 : i64}
    %1 = cir.stack_save : !cir.ptr<!u8i>
    cir.store %1, %0 : !cir.ptr<!u8i>, cir.ptr <!cir.ptr<!u8i>>
    %2 = cir.load %0 : cir.ptr <!cir.ptr<!u8i>>, !cir.ptr<!u8i>
    cir.stack_restore %2 : !cir.ptr<!u8i>
    cir.return
  }
}

//      MLIR: module {
// MLIR-NEXT:   llvm.func @foo() 
// MLIR-NEXT:     %0 = llvm.mlir.constant(1 : index) : i64
// MLIR-NEXT:     %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i64) -> !llvm.ptr
// MLIR-NEXT:     %2 = llvm.intr.stacksave : !llvm.ptr
// MLIR-NEXT:     llvm.store %2, %1 : !llvm.ptr, !llvm.ptr
// MLIR-NEXT:     %3 = llvm.load %1 : !llvm.ptr -> !llvm.ptr
// MLIR-NEXT:     llvm.intr.stackrestore %3 : !llvm.ptr
// MLIR-NEXT:     llvm.return
// MLIR-NEXT:   }
// MLIR-NEXT: }