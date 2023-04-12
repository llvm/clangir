// RUN: %clang -fclangir-enable -fclangir-direct-lowering -o %t %s
// RUN: %t | FileCheck %s
int printf(const char *format);

int main (void) {
    printf ("Hello, world!\n");
    // CHECK: Hello, world!
    return 0;
}
