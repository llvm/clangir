// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -std=c++17 %s -o - | FileCheck %s

void bar(const int &i = 42);

void foo() {
  bar();
}

// CHECK: %0 = cir.alloca !s32i
// CHECK: %1 = cir.const #cir.int<42>
// CHECK: cir.store %1, %0
// CHECK: cir.call @_Z3barRKi(%0)

