// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-threadsafe-statics -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
//
// Test static local variable with non-trivial destructor
// (Previously crashed with "C++ guarded init is NYI")

class a {
public:
  ~a();
};
void b() { static a c; }

// CHECK: cir.global "private" internal dso_local @_ZZ1bvE1c = #cir.zero : !rec_a
// CHECK: cir.global "private" internal dso_local @_ZGVZ1bvE1c = #cir.int<0> : !u8i
// CHECK: cir.func {{.*}} @_Z1bv()
// CHECK: cir.get_global @_ZZ1bvE1c
// CHECK: cir.get_global @_ZGVZ1bvE1c
// CHECK: cir.load
// CHECK: cir.cmp(eq,
// CHECK: cir.if
// CHECK: cir.return
