// RUN: split-file %s %t
// RUN: %clang_cc1 -fclangir -fno-clangir-direct-lowering -emit-mlir=core %t%{fs-sep}test_declaration.c 
// RUN: FileCheck --input-file=%t%{fs-sep}test_declaration.mlir %t%{fs-sep}test_declaration.c


//--- add10.h

#ifndef ADD10_H
#define ADD10_H

int add10(int x);

#endif


//--- test_declaration.c

#include "add10.h"

// CHECK: func.func private @add10(i32) -> i32

int main(int argc, char *argv[]) {
    // Variables
    int number = 15;

    number = add10(number);

    return 0;
}