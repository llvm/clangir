// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
//
// Bitfield bool to int conversion - load operation test
// Tests that loading a bool bitfield correctly uses an integer type
// for the get_bitfield operation, then casts to bool.
// Note: Stores to bool bitfields are NYI and will crash.

struct a {
  bool b : 1;
};
class c {
public:
  void operator<<(int);
};

// CHECK: cir.func {{.*@_Z1d1c1a}}
// CHECK:   %[[MEMBER:.*]] = cir.get_member %{{.*}}[0] {name = "b"} : !cir.ptr<!{{.*}}> -> !cir.ptr<!u8i>
// CHECK:   %[[BITFIELD:.*]] = cir.get_bitfield{{.*}}%[[MEMBER]] : !cir.ptr<!u8i>) -> <u, 1>
// CHECK:   %[[BOOL:.*]] = cir.cast int_to_bool %[[BITFIELD]] : !cir.int<u, 1> -> !cir.bool
// CHECK:   %[[INT:.*]] = cir.cast bool_to_int %[[BOOL]] : !cir.bool -> !s32i
// CHECK:   cir.call @_ZN1clsEi(%{{.*}}, %[[INT]])

// LLVM: define {{.*}}@_Z1d1c1a
// LLVM:   %[[LOAD:.*]] = load i8, ptr
// LLVM:   %[[AND:.*]] = and i8 %[[LOAD]], 1
// LLVM:   %[[TRUNC:.*]] = trunc i8 %[[AND]] to i1
// LLVM:   %[[ZEXT:.*]] = zext i1 %{{.*}} to i32
// LLVM:   call {{.*}}@_ZN1clsEi

void d(c e, a f) { e << f.b; }
