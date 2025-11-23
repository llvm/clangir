// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-threadsafe-statics -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
//
// Test static local variable with template constructor
// (Previously crashed with assertion failure in dyn_cast)

struct a {
  template <typename b, typename c> a(b, c);
};
class d {
  a e;

public:
  d(int) : e(0, 0) {}
};
void f() { static d g(0); }

// CHECK: cir.global "private" internal dso_local @_ZZ1fvE1g = #cir.zero : !rec_d
// CHECK: cir.global "private" internal dso_local @_ZGVZ1fvE1g = #cir.int<0> : !u8i
// CHECK: cir.func dso_local @_Z1fv()
// CHECK: cir.get_global @_ZZ1fvE1g
// CHECK: cir.get_global @_ZGVZ1fvE1g
// CHECK: cir.load
// CHECK: cir.cmp(eq,
// CHECK: cir.if
// CHECK: cir.call @_ZN1dC1Ei
// CHECK: cir.return
