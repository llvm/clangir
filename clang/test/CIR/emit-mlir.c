// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o - | FileCheck %s -check-prefix=STD

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=std %s -o - 2>&1 | FileCheck %s -check-prefix=STD_ERR

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir=llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir=std %s -o - | FileCheck %s -check-prefix=STD

// RUN: %clang -fclangir -Xclang -emit-mlir %s -o - -### 2>&1 | FileCheck %s -check-prefix=OPTS_NO_VALUE
// RUN: %clang -fclangir -Xclang -emit-mlir=llvm %s -o - -###  2>&1 | FileCheck %s -check-prefix=OPTS_LLVM
// RUN: %clang -fno-clangir-direct-lowering -Xclang -emit-mlir=std %s -o - -### 2>&1 | FileCheck %s -check-prefix=OPTS_STD

int foo(int a, int b) {
    return a + b;
}

// LLVM: llvm.func @foo
// STD: func.func @foo
// STD_ERR: ClangIR direct lowering is incompatible with emitting of MLIR standard dialects
// OPTS_NO_VALUE: "-emit-mlir"
// OPTS_LLVM: "-emit-mlir=llvm"
// OPTS_STD: "-emit-mlir=std"
