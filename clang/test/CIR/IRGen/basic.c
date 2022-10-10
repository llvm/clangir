// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

void foo() {}

//      CHECK: define void @foo()
// CHECK-NEXT:   ret void,
// CHECK-NEXT: }
