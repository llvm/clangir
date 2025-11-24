// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
//
// Bitfield bool load and store operations test
// Tests that loading and storing bool bitfields correctly use integer types
// with explicit casts between BoolType and IntType.

struct a {
  bool b : 1;
};
class c {
public:
  void operator<<(int);
};

// Test loading bool bitfield
// CHECK-LABEL: cir.func {{.*@_Z1d1c1a}}
// CHECK:   %[[MEMBER:.*]] = cir.get_member %{{.*}}[0] {name = "b"} : !cir.ptr<!{{.*}}> -> !cir.ptr<!u8i>
// CHECK:   %[[BITFIELD:.*]] = cir.get_bitfield{{.*}}%[[MEMBER]] : !cir.ptr<!u8i>) -> <u, 1>
// CHECK:   %[[BOOL:.*]] = cir.cast int_to_bool %[[BITFIELD]] : !cir.int<u, 1> -> !cir.bool
// CHECK:   %[[INT:.*]] = cir.cast bool_to_int %[[BOOL]] : !cir.bool -> !s32i
// CHECK:   cir.call @_ZN1clsEi(%{{.*}}, %[[INT]])

// LLVM-LABEL: define {{.*}}@_Z1d1c1a
// LLVM:   %[[LOAD:.*]] = load i8, ptr
// LLVM:   %[[AND:.*]] = and i8 %[[LOAD]], 1
// LLVM:   %[[TRUNC:.*]] = trunc i8 %[[AND]] to i1
// LLVM:   %[[ZEXT:.*]] = zext i1 %{{.*}} to i32
// LLVM:   call {{.*}}@_ZN1clsEi

void d(c e, a f) { e << f.b; }

// Test storing constant bool values to bitfield
// CHECK-LABEL: cir.func {{.*@_Z10store_testR1a}}
// CHECK:   cir.const #true
// CHECK:   cir.cast bool_to_int %{{.*}} : !cir.bool -> !cir.int<u, 1>
// CHECK:   cir.set_bitfield{{.*}} -> <u, 1>

// CHECK:   cir.const #false
// CHECK:   cir.cast bool_to_int %{{.*}} : !cir.bool -> !cir.int<u, 1>
// CHECK:   cir.set_bitfield{{.*}} -> <u, 1>

// LLVM-LABEL: define {{.*}}@_Z10store_testR1a
// LLVM:   %[[LOAD1:.*]] = load i8, ptr
// LLVM:   %[[AND1:.*]] = and i8 %[[LOAD1]], -2
// LLVM:   %[[OR1:.*]] = or i8 %[[AND1]], 1
// LLVM:   store i8 %[[OR1]], ptr

// LLVM:   %[[LOAD2:.*]] = load i8, ptr
// LLVM:   %[[AND2:.*]] = and i8 %[[LOAD2]], -2
// LLVM:   %[[OR2:.*]] = or i8 %[[AND2]], 0
// LLVM:   store i8 %[[OR2]], ptr

void store_test(a& x) {
  x.b = true;
  x.b = false;
}

// Test storing computed bool values to bitfield
// CHECK-LABEL: cir.func {{.*@_Z19store_computed_testR1abb}}
// CHECK:   cir.cast bool_to_int %{{.*}} : !cir.bool -> !s32i
// CHECK:   cir.cast bool_to_int %{{.*}} : !cir.bool -> !s32i
// CHECK:   cir.binop(and, %{{.*}}, %{{.*}}) : !s32i
// CHECK:   cir.cast int_to_bool %{{.*}} : !s32i -> !cir.bool
// CHECK:   cir.cast bool_to_int %{{.*}} : !cir.bool -> !cir.int<u, 1>
// CHECK:   cir.set_bitfield{{.*}} -> <u, 1>

// LLVM-LABEL: define {{.*}}@_Z19store_computed_testR1abb
// LLVM:   and i8 %{{.*}}, 1
// LLVM:   and i8 %{{.*}}, -2
// LLVM:   or i8 %{{.*}}, %{{.*}}
// LLVM:   store i8 %{{.*}}, ptr

void store_computed_test(a& x, bool v1, bool v2) {
  x.b = v1 & v2;
}

// Test load-store round-trip
// CHECK-LABEL: cir.func {{.*@_Z14roundtrip_testR1aS0_}}
// CHECK:   cir.get_bitfield{{.*}} -> <u, 1>
// CHECK:   cir.cast int_to_bool %{{.*}} : !cir.int<u, 1> -> !cir.bool
// CHECK:   cir.cast bool_to_int %{{.*}} : !cir.bool -> !cir.int<u, 1>
// CHECK:   cir.set_bitfield{{.*}} -> <u, 1>

// LLVM-LABEL: define {{.*}}@_Z14roundtrip_testR1aS0_
// LLVM:   %[[SRC_LOAD:.*]] = load i8, ptr
// LLVM:   %[[SRC_AND:.*]] = and i8 %[[SRC_LOAD]], 1
// LLVM:   %[[DST_LOAD:.*]] = load i8, ptr
// LLVM:   %[[DST_CLEAR:.*]] = and i8 %[[DST_LOAD]], -2
// LLVM:   %[[DST_SET:.*]] = or i8 %[[DST_CLEAR]],
// LLVM:   store i8 %[[DST_SET]], ptr

void roundtrip_test(a& src, a& dst) {
  dst.b = src.b;
}
