// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// Test basic file-scope assembly
asm(".section .text");
asm(".global my_func");
asm("my_func: ret");

// Test file-scope assembly with functions
int foo() { 
  return 42; 
}

asm(".data");
asm(".align 8");

int bar() { 
  return 24; 
}

// Test file-scope assembly at end of file
asm(".section .rodata");
asm(".string \"hello\"");

int main() {
  return foo() + bar();
}

// CIR: cir.module_asm = #cir.module_asm<[".section .text", ".global my_func", "my_func: ret", ".data", ".align 8", ".section .rodata", ".string {{\\22}}hello{{\\22}}"

// CIR: cir.func {{.*}}@foo
// CIR: cir.func {{.*}}@bar
// CIR: cir.func {{.*}}@main

// LLVM: module asm ".section .text"
// LLVM: module asm ".global my_func"
// LLVM: module asm "my_func: ret"
// LLVM: module asm ".data"
// LLVM: module asm ".align 8"
// LLVM: module asm ".section .rodata"
// LLVM: module asm ".string {{\\22}}hello{{\\22}}"

// LLVM: define{{.*}} i32 @foo()
// LLVM: define{{.*}} i32 @bar()
// LLVM: define{{.*}} i32 @main()

