// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

using i10 = signed _BitInt(10);
using u10 = unsigned _BitInt(10);

i10 test_signed(i10 arg) {
  return arg;
}

// CHECK: cir.func @_Z11test_signedDB10_(%arg0: !cir.int<s, 10> loc({{.*}}) -> !cir.int<s, 10>
// CHECK: }

u10 test_unsigned(u10 arg) {
  return arg;
}

// CHECK: cir.func @_Z13test_unsignedDU10_(%arg0: !cir.int<u, 10> loc({{.*}}) -> !cir.int<u, 10>
// CHECK: }

i10 test_init() {
  return 42;
}

//      CHECK: cir.func @_Z9test_initv() -> !cir.int<s, 10>
//      CHECK:   %[[#LITERAL:]] = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:   %{{.+}} = cir.cast(integral, %[[#LITERAL]] : !s32i), !cir.int<s, 10>
//      CHECK: }

void test_init_for_mem() {
  i10 x = 42;
}

//      CHECK: cir.func @_Z17test_init_for_memv()
//      CHECK:   %[[#LITERAL:]] = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK-NEXT:   %[[#INIT:]] = cir.cast(integral, %[[#LITERAL]] : !s32i), !cir.int<s, 10>
// CHECK-NEXT:   cir.store %[[#INIT]], %{{.+}} : !cir.int<s, 10>, cir.ptr <!cir.int<s, 10>>
//      CHECK: }

i10 test_arith(i10 lhs, i10 rhs) {
  return lhs + rhs;
}

//      CHECK: cir.func @_Z10test_arithDB10_S_(%arg0: !cir.int<s, 10> loc({{.+}}), %arg1: !cir.int<s, 10> loc({{.+}})) -> !cir.int<s, 10>
//      CHECK:   %[[#LHS:]] = cir.load %{{.+}} : cir.ptr <!cir.int<s, 10>>, !cir.int<s, 10>
// CHECK-NEXT:   %[[#RHS:]] = cir.load %{{.+}} : cir.ptr <!cir.int<s, 10>>, !cir.int<s, 10>
// CHECK-NEXT:   %{{.+}} = cir.binop(add, %[[#LHS]], %[[#RHS]]) : !cir.int<s, 10>
//      CHECK: }
