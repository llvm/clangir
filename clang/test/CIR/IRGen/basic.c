// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-obj %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s -check-prefix=OBJ

void foo() {}

//      LLVM: define void @foo()
// LLVM-NEXT:   ret void,
// LLVM-NEXT: }

// OBJ: 0: c3 retq
