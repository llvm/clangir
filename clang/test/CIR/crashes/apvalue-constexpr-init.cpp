// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
//
// Test for APValue emission for complex constexpr initializers with:
// - Arrays of structs
// - Member function pointers
// - Nested aggregate initialization
//
// Originally reduced from /tmp/MSP430AttributeParser-875bbc.cpp

template <typename a, int b> struct c {
  typedef a d[b];
};
template <typename a, int b> struct h {
  typename c<a, b>::d e;
};
enum f { g };
class i {
  struct m {
    f j;
    int (i::*k)(f);
  };
  static const h<m, 4> l;
  int n(f);
};

// CHECK: cir.global constant weak_odr comdat @_ZN1i1lE = #cir.const_record<{
// CHECK-SAME: #cir.const_array<[
// CHECK-SAME: #cir.const_record<{#cir.int<0> : !u32i, #cir.method<@_ZN1i1nE1f>
// CHECK-SAME: #cir.zero
// CHECK-SAME: #cir.zero
// CHECK-SAME: #cir.zero
constexpr h<i::m, 4> i::l{g, &i::n, {}, {}, {}};
