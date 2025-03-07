// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core-flat %s -o - | FileCheck %s

void foo() {
  int a = 0;
  int b = 0;
  if (a < 0)
    return;
  ++b;
  return;
}

// CHECK-LABEL: func.func @foo() {
// CHECK:    %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK:    %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK:    %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK:    memref.store %[[c0_i32]], %[[alloca]][] : memref<i32>
// CHECK:    %[[c0_i32_1:.+]] = arith.constant 0 : i32
// CHECK:    memref.store %[[c0_i32_1]], %[[alloca_0]][] : memref<i32>
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32>
// CHECK:    %[[c0_i32_2:.+]] = arith.constant 0 : i32
// CHECK:    %[[ONE:.+]] = arith.cmpi slt, %[[ZERO]], %[[c0_i32_2]] : i32
// CHECK:    cf.cond_br %[[ONE]], ^bb2, ^bb3
// CHECK:  ^bb2:
// CHECK:    cf.br ^bb5
// CHECK:  ^bb3:
// CHECK:    cf.br ^bb4
// CHECK:  ^bb4:
// CHECK:    %[[TWO:.+]] = memref.load %[[alloca_0]][] : memref<i32>
// CHECK:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK:    %[[THREE:.+]] = arith.addi %[[TWO]], %[[c1_i32]] : i32
// CHECK:    memref.store %[[THREE]], %[[alloca_0]][] : memref<i32>
// CHECK:    cf.br ^bb5
// CHECK:  ^bb5:
// CHECK:    return
// CHECK:  }

void foo2() {
  int b = 0;
  for (int i = 0; i < 8; ++i)
    if (b > 4)
      return;
  ++b;
  return;
}

// CHECK-LABEL:   func.func @foo2() {
// CHECK:     %[[alloca:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK:     %[[alloca_0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK:     memref.store %[[c0_i32]], %[[alloca_0]][] : memref<i32>
// CHECK:     cf.br ^bb1
// CHECK:   ^bb1:
// CHECK:     %[[c0_i32_1:.+]] = arith.constant 0 : i32
// CHECK:     memref.store %[[c0_i32_1]], %[[alloca]][] : memref<i32>
// CHECK:     cf.br ^bb2
// CHECK:   ^bb2:
// CHECK:     %[[ZERO:.+]] = memref.load %[[alloca]][] : memref<i32>
// CHECK:     %[[c8_i32:.+]] = arith.constant 8 : i32
// CHECK:     %[[ONE:.+]] = arith.cmpi slt, %[[ZERO]], %[[c8_i32]] : i32
// CHECK:     cf.cond_br %[[ONE]], ^bb3, ^bb9
// CHECK:   ^bb3:
// CHECK:     cf.br ^bb4
// CHECK:   ^bb4:
// CHECK:     %[[TWO:.+]] = memref.load %[[alloca_0]][] : memref<i32>
// CHECK:     %[[c4_i32:.+]] = arith.constant 4 : i32
// CHECK:     %[[THREE:.+]] = arith.cmpi sgt, %[[TWO]], %[[c4_i32]] : i32
// CHECK:     cf.cond_br %[[THREE]], ^bb5, ^bb6
// CHECK:   ^bb5:
// CHECK:     cf.br ^bb11
// CHECK:   ^bb6:
// CHECK:     cf.br ^bb7
// CHECK:   ^bb7:
// CHECK:     cf.br ^bb8
// CHECK:   ^bb8:
// CHECK:     %[[FOUR:.+]] = memref.load %[[alloca]][] : memref<i32>
// CHECK:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK:     %[[FIVE:.+]] = arith.addi %[[FOUR]], %[[c1_i32]] : i32
// CHECK:     memref.store %[[FIVE]], %[[alloca]][] : memref<i32>
// CHECK:     cf.br ^bb2
// CHECK:   ^bb9:
// CHECK:     cf.br ^bb10
// CHECK:   ^bb10:
// CHECK:     %[[SIX:.+]] = memref.load %[[alloca_0]][] : memref<i32>
// CHECK:     %[[c1_i32_2:.+]] = arith.constant 1 : i32
// CHECK:     %[[SEVEN:.+]] = arith.addi %[[SIX]], %[[c1_i32_2]] : i32
// CHECK:     memref.store %[[SEVEN]], %[[alloca_0]][] : memref<i32>
// CHECK:     cf.br ^bb11
// CHECK:   ^bb11:
// CHECK:     return
// CHECK:   }
