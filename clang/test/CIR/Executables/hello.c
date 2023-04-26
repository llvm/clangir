// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir-enable -fclangir-direct-lowering -o %t %s
// RUN: %t | FileCheck %s
// REQUIRES: system-linux
// REQUIRES: target-linux
int printf(const char *format);

int main (void) {
    printf ("Hello, world!\n");
    // CHECK: Hello, world!
    return 0;
}
