// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

void foo() {
  double bar[] = {9,8,7};
}

//      CHECK: %0 = cir.alloca !cir.array<!cir.double x 3>, cir.ptr <!cir.array<!cir.double x 3>>, ["bar"] {alignment = 16 : i64}
// CHECK-NEXT: %1 = cir.const(#cir.const_array<[#cir.float<9.000000e+00> : !cir.double, #cir.float<8.000000e+00> : !cir.double, #cir.float<7.000000e+00> : !cir.double]> : !cir.array<!cir.double x 3>) : !cir.array<!cir.double x 3>
// CHECK-NEXT: cir.store %1, %0 : !cir.array<!cir.double x 3>, cir.ptr <!cir.array<!cir.double x 3>>
