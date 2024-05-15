// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func {{@.*foo.*}}(%arg0: !cir.ptr<!s32i>
void foo(int __attribute__((address_space(0))) *arg) {
  return;
}

// CHECK: cir.func {{@.*bar.*}}(%arg0: !cir.ptr<!s32i, addrspace(1)>
void bar(int __attribute__((address_space(1))) *arg) {
  return;
}
