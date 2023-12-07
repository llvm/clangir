// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o - | FileCheck %s

typedef int int4 __attribute__((vector_size(16)));
int main() {
  int4 a = { 1, 2, 3, 4 };
  int4 b = { 5, 6, 7, 8 };
  int4 c = a + b;
  return c[1];
}

// CHECK:    %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    %1 = cir.alloca !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>, ["a", init] {alignment = 16 : i64}
// CHECK:    %2 = cir.alloca !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>, ["b", init] {alignment = 16 : i64}
// CHECK:    %3 = cir.alloca !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>, ["c", init] {alignment = 16 : i64}
// CHECK:    %4 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:    %5 = cir.const(#cir.int<2> : !s32i) : !s32i
// CHECK:    %6 = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK:    %7 = cir.const(#cir.int<4> : !s32i) : !s32i
// CHECK:    %8 = cir.vec(%4, %5, %6, %7 : !s32i, !s32i, !s32i, !s32i) : <!s32i x 4>
// CHECK:    cir.store %8, %1 : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>
// CHECK:    %9 = cir.const(#cir.int<5> : !s32i) : !s32i
// CHECK:    %10 = cir.const(#cir.int<6> : !s32i) : !s32i
// CHECK:    %11 = cir.const(#cir.int<7> : !s32i) : !s32i
// CHECK:    %12 = cir.const(#cir.int<8> : !s32i) : !s32i
// CHECK:    %13 = cir.vec(%9, %10, %11, %12 : !s32i, !s32i, !s32i, !s32i) : <!s32i x 4>
// CHECK:    cir.store %13, %2 : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>
// CHECK:    %14 = cir.load %1 : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK:    %15 = cir.load %2 : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK:    %16 = cir.binop(add, %14, %15) : !cir.vector<!s32i x 4>
// CHECK:    cir.store %16, %3 : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>
// CHECK:    %17 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:    %18 = cir.load %3 : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK:    %19 = cir.vec_elem %18[%17 : !s32i] <!s32i x 4> -> !s32i
// CHECK:    cir.store %19, %0 : !s32i, cir.ptr <!s32i>
// CHECK:    %20 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK:    cir.return %20 : !s32i
