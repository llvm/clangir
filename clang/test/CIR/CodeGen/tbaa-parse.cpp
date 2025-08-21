// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -o %t -O1
// RUN: cir-opt %t

struct S {
    short i;
};

struct S glob;
int main(void) {
    glob.i = 0;
    return glob.i;
}
