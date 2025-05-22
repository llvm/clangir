// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir 
// RUN: FileCheck --input-file=%t.mlir %s


int declaration(int x);

// CHECK: func.func private @declaration(i32) -> i32

int declaration_test() {
    // Variables
    int number = 15;

    number = declaration(number);

    return 0;
}