// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o - | FileCheck %s

typedef int int4 __attribute__((vector_size(16)));
int main() {
  int4 a = { 1, 2, 3, 4 };
  int4 b = { 5, 6, 7, 8 };
  int4 c = a + b;
  return c[1];
}

// CHECK:    %{{[0-9]+}} = cir.vec(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : !s32i, !s32i, !s32i, !s32i) : <!s32i x 4>
// CHECK:    cir.store %{{[0-9]+}}, %{{[0-9]+}} : !cir.vector<!s32i x 4>, cir.ptr <!cir.vector<!s32i x 4>>
// CHECK:    %{{[0-9]+}} = cir.load %{{[0-9]+}} : cir.ptr <!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>
// CHECK:    %{{[0-9]+}} = cir.binop(add, %{{[0-9]+}}, %{{[0-9]+}}) : !cir.vector<!s32i x 4>
// CHECK:    %{{[0-9]+}} = cir.vec_elem %{{[0-9]+}}[%{{[0-9]+}} : !s32i] <!s32i x 4> -> !s32i
