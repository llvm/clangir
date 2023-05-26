// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s -check-prefix=MLIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -S %s -o %t.s
// RUN: FileCheck --input-file=%t.s %s -check-prefix=ASM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-obj %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s -check-prefix=OBJ

void foo() {}

//      MLIR: func.func @foo() {
// MLIR-NEXT:   return
// MLIR-NEXT: }

//      LLVM: define void @foo()
// LLVM-NEXT:   ret void
// LLVM-NEXT: }

//      ASM: .globl  foo
// ASM-NEXT: .p2align
// ASM-NEXT: .type foo,@function
// ASM-NEXT: foo:
//      ASM: retq

// OBJ: 0: c3 retq
