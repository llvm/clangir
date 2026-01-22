// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

int literals(void) {
    char a = 'a'; // char literals are int in C, but CIR folds the cast
    // CHECK: cir.const #cir.int<97> : !s8i

    return 0;
}
