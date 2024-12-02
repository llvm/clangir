// RUN: %clang_cc1  -triple aarch64-none-linux-android21  -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void foo(int *iptr) { iptr + 2; }

// LLVM: getelementptr inbounds 