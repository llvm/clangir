// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -Wno-return-stack-address -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int main(void) {
    char a = 'a'; // char literals have char type in C++
    // CHECK:  %{{[0-9]+}} = cir.const(97 : i8) : i8

    return 0;
}
