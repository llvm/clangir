// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int test_load(volatile int *ptr) {
  return *ptr;
}

//      CHECK: cir.func @_Z9test_loadPVi
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:   %2 = cir.load deref %0 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   %3 = cir.load volatile %2 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.store %3, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.return %4 : !s32i

void test_store(volatile int *ptr) {
  *ptr = 42;
}

//      CHECK: cir.func @_Z10test_storePVi
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:   %1 = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:   %2 = cir.load deref %0 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store volatile %1, %2 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.return

struct Foo {
  int x;
  volatile int y;
  volatile int z: 4;
};

int test_load_field1(volatile Foo *ptr) {
  return ptr->x;
}

//      CHECK: cir.func @_Z16test_load_field1PV3Foo
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Foo22>>, !cir.ptr<!ty_22Foo22>
// CHECK-NEXT:   %3 = cir.get_member %2[0] {name = "x"} : !cir.ptr<!ty_22Foo22> -> !cir.ptr<!s32i>
// CHECK-NEXT:   %4 = cir.load volatile %3 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.store %4, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %5 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.return %5 : !s32i

int test_load_field2(Foo *ptr) {
  return ptr->y;
}

//      CHECK: cir.func @_Z16test_load_field2P3Foo
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Foo22>>, !cir.ptr<!ty_22Foo22>
// CHECK-NEXT:   %3 = cir.get_member %2[1] {name = "y"} : !cir.ptr<!ty_22Foo22> -> !cir.ptr<!s32i>
// CHECK-NEXT:   %4 = cir.load volatile %3 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.store %4, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %5 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.return %5 : !s32i

int test_load_field3(Foo *ptr) {
  return ptr->z;
}

//      CHECK: cir.func @_Z16test_load_field3P3Foo
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %1 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Foo22>>, !cir.ptr<!ty_22Foo22>
// CHECK-NEXT:   %3 = cir.get_member %2[2] {name = "z"} : !cir.ptr<!ty_22Foo22> -> !cir.ptr<!u8i>
// CHECK-NEXT:   %4 = cir.load volatile %3 : cir.ptr <!u8i>, !u8i
// CHECK-NEXT:   %5 = cir.cast(integral, %4 : !u8i), !s8i
// CHECK-NEXT:   %6 = cir.const(#cir.int<4> : !s8i) : !s8i
// CHECK-NEXT:   %7 = cir.shift(left, %5 : !s8i, %6 : !s8i) -> !s8i
// CHECK-NEXT:   %8 = cir.const(#cir.int<4> : !s8i) : !s8i
// CHECK-NEXT:   %9 = cir.shift( right, %7 : !s8i, %8 : !s8i) -> !s8i
// CHECK-NEXT:   %10 = cir.cast(integral, %9 : !s8i), !s32i
// CHECK-NEXT:   cir.store %10, %1 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   %11 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.return %11 : !s32i

void test_store_field1(volatile Foo *ptr) {
  ptr->x = 42;
}

//      CHECK: cir.func @_Z17test_store_field1PV3Foo
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %1 = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Foo22>>, !cir.ptr<!ty_22Foo22>
// CHECK-NEXT:   %3 = cir.get_member %2[0] {name = "x"} : !cir.ptr<!ty_22Foo22> -> !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store volatile %1, %3 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.return

void test_store_field2(Foo *ptr) {
  ptr->y = 42;
}

//      CHECK: cir.func @_Z17test_store_field2P3Foo
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %1 = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Foo22>>, !cir.ptr<!ty_22Foo22>
// CHECK-NEXT:   %3 = cir.get_member %2[1] {name = "y"} : !cir.ptr<!ty_22Foo22> -> !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store volatile %1, %3 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:   cir.return

void test_store_field3(Foo *ptr) {
  ptr->z = 4;
}

//      CHECK: cir.func @_Z17test_store_field3P3Foo
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   cir.store %arg0, %0 : !cir.ptr<!ty_22Foo22>, cir.ptr <!cir.ptr<!ty_22Foo22>>
// CHECK-NEXT:   %1 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK-NEXT:   %2 = cir.load %0 : cir.ptr <!cir.ptr<!ty_22Foo22>>, !cir.ptr<!ty_22Foo22>
// CHECK-NEXT:   %3 = cir.get_member %2[2] {name = "z"} : !cir.ptr<!ty_22Foo22> -> !cir.ptr<!u8i>
// CHECK-NEXT:   %4 = cir.cast(integral, %1 : !s32i), !u8i
// CHECK-NEXT:   %5 = cir.load %3 : cir.ptr <!u8i>, !u8i
// CHECK-NEXT:   %6 = cir.const(#cir.int<15> : !u8i) : !u8i
// CHECK-NEXT:   %7 = cir.binop(and, %4, %6) : !u8i
// CHECK-NEXT:   %8 = cir.const(#cir.int<240> : !u8i) : !u8i
// CHECK-NEXT:   %9 = cir.binop(and, %5, %8) : !u8i
// CHECK-NEXT:   %10 = cir.binop(or, %9, %7) : !u8i
// CHECK-NEXT:   cir.store volatile %10, %3 : !u8i, cir.ptr <!u8i>
// CHECK-NEXT:   cir.return
