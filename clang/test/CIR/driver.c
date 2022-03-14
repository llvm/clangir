// RUN: %clang -target x86_64-unknown-linux-gnu -fenable-clangir -S -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang -target x86_64-unknown-linux-gnu -fenable-clangir -S -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang -target x86_64-unknown-linux-gnu -fenable-clangir -c %s -o %t.o
// RUN: llvm-objdump -d %t.o | FileCheck %s -check-prefix=OBJ
// RUN: %clang -target x86_64-unknown-linux-gnu -fenable-clangir -disable-cir-passes -S -emit-cir %s -o %t.cir
// RUN: %clang -target arm64-apple-macosx12.0.0 -fenable-clangir -S -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// XFAIL: *

void foo() {}

//      CIR: module  {
// CIR-NEXT:   func @foo() {
// CIR-NEXT:     cir.return
// CIR-NEXT:   }
// CIR-NEXT: }

//      LLVM: define void @foo()
// LLVM-NEXT:   ret void,
// LLVM-NEXT: }

// OBJ: 0: c3 retq
