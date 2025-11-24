// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
//
// Test for member pointer base-to-derived casts in constant expressions.
// This was previously failing but is now fixed by the array handling and
// MethodType::getABIAlignment fixes.

class a {
public:
  int b(unsigned);
};
class c : a {
  struct d {
    int (c::*e)(unsigned);
  } static const f[];
};

// CHECK: cir.global constant external @_ZN1c1fE = #cir.const_array<[
// CHECK-SAME: #cir.const_record<{#cir.method<@_ZN1a1bEj>
const c::d c::f[]{&a::b};
