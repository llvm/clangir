// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -S -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// CIR: cir.func {{@.*foo.*}}(%arg0: !cir.ptr<!s32i>
// LLVM: define void @foo(ptr %0)
void foo(int __attribute__((address_space(0))) *arg) {
  return;
}

// CIR: cir.func {{@.*bar.*}}(%arg0: !cir.ptr<!s32i, addrspace(1)>
// LLVM: define void @bar(ptr addrspace(1) %0)
void bar(int __attribute__((address_space(1))) *arg) {
  return;
}
