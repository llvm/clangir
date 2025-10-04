// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s < %t.cir

struct Wrapper {
  static inline const int &ref = 7;
};

const int *addr() { return &Wrapper::ref; }

// CHECK: cir.global constant linkonce_odr comdat @_ZN7Wrapper3refE = #cir.global_view<@_ZGRN7Wrapper3refE_> : !cir.ptr<!s32i>
// CHECK: cir.global linkonce_odr comdat @_ZGRN7Wrapper3refE_ = #cir.int<7> : !s32i
