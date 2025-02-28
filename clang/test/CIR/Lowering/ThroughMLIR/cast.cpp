// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s


int main() {
  int a[10];
  // CHECK: %[[ALLOCA_0:.+]] = memref.alloca() {alignment = 16 : i64} : memref<10xi32>
  int b[4][7];
  // CHECK: %[[ALLOCA_1:.+]] = memref.alloca() {alignment = 16 : i64} : memref<4x7xi32>

int *aa = a;
  // CHECK: %[[ALLOCA_2:.+]] = memref.alloca() {alignment = 8 : i64} : memref<memref<i32>>
  // CHECK: %[[REINTERPRET:.+]] = memref.reinterpret_cast %[[ALLOCA_0]] to offset: [0], sizes: [], strides: [] : memref<10xi32> to memref<i32>
  // CHECK: memref.store %[[REINTERPRET:.+]], %[[ALLOCA_2:.+]][] : memref<memref<i32>

  int *p = &a[0];
  // CHECK: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK: %[[REINTERPRET_11:.+]] = memref.reinterpret_cast %[[ALLOCA_0]] to offset: [0], sizes: [], strides: [] : memref<10xi32> to memref<i32>
  // CHECK: %[[C0_i:.+]] = arith.index_cast %[[C0]] : i32 to index
  // CHECK: %[[REINTERPRET_12:.+]] = memref.reinterpret_cast %[[REINTERPRET_11]] to offset: [%0], sizes: [], strides: [] : memref<i32> to memref<i32, strided<[], offset: ?>>
  // CHECK: %[[CAST:.+]] = memref.cast %[[REINTERPRET_12]] : memref<i32, strided<[], offset: ?>> to memref<i32>
  // CHECK: memref.store %[[CAST]], %[[ALLOCA_3:.+]][] : memref<memref<i32>>
  auto *ap = &a;
  *ap[3] = 7;

  auto *ap10 = (int (*)[10]) p;
  // CHECK: %[[p3:.+]] = memref.load %[[ALLOCA_3]][] : memref<memref<i32>>
  // CHECK: %[[REINTERPRET_16:.+]] = memref.reinterpret_cast %[[p3]] to offset: [0], sizes: [10], strides: [1] : memref<i32> to memref<10xi32>
  // CHECK: memref.store %[[REINTERPRET_16]], %[[ALLOCA_5:.+]][] : memref<memref<10xi32>>

  auto v15 = b[1][5];
  int *bpd = b[0];
  auto pb36 = &b[3][6];
  auto pb2 = &b[2];
  auto pb2a = b[2];

  return a[3];
}
/*  func.func @main() -> i32 {
    %alloca_3 = memref.alloca() {alignment = 8 : i64} : memref<memref<i32>> loc(#loc54)
    %alloca_4 = memref.alloca() {alignment = 8 : i64} : memref<memref<10xi32>> loc(#loc55)
    %alloca_5 = memref.alloca() {alignment = 8 : i64} : memref<memref<10xi32>> loc(#loc56)
    %alloca_6 = memref.alloca() {alignment = 4 : i64} : memref<i32> loc(#loc57)
    %alloca_7 = memref.alloca() {alignment = 8 : i64} : memref<memref<i32>> loc(#loc58)
    %alloca_8 = memref.alloca() {alignment = 8 : i64} : memref<memref<i32>> loc(#loc59)
    %alloca_9 = memref.alloca() {alignment = 8 : i64} : memref<memref<7xi32>> loc(#loc60)
    %alloca_10 = memref.alloca() {alignment = 8 : i64} : memref<memref<i32>> loc(#loc61)

    memref.store %alloca_0, %alloca_4[] : memref<memref<10xi32>> loc(#loc55)
    %c7_i32 = arith.constant 7 : i32 loc(#loc27)
    %1 = memref.load %alloca_4[] : memref<memref<10xi32>> loc(#loc28)
    %c3_i32 = arith.constant 3 : i32 loc(#loc29)
    %2 = arith.index_cast %c3_i32 : i32 to index loc(#loc30)
    %reinterpret_cast_13 = memref.reinterpret_cast %1 to offset: [%2], sizes: [10], strides: [1] : memref<10xi32> to memref<10xi32, strided<[1], offset: ?>> loc(#loc30)
    %cast_14 = memref.cast %reinterpret_cast_13 : memref<10xi32, strided<[1], offset: ?>> to memref<10xi32> loc(#loc30)
    %reinterpret_cast_15 = memref.reinterpret_cast %cast_14 to offset: [0], sizes: [], strides: [] : memref<10xi32> to memref<i32> loc(#loc62)
    memref.store %c7_i32, %reinterpret_cast_15[] : memref<i32> loc(#loc63)
    %c1_i32 = arith.constant 1 : i32 loc(#loc32)
    %reinterpret_cast_17 = memref.reinterpret_cast %alloca_1 to offset: [0], sizes: [7], strides: [1] : memref<4x7xi32> to memref<7xi32> loc(#loc33)
    %4 = arith.index_cast %c1_i32 : i32 to index loc(#loc33)
    %reinterpret_cast_18 = memref.reinterpret_cast %reinterpret_cast_17 to offset: [%4], sizes: [7], strides: [1] : memref<7xi32> to memref<7xi32, strided<[1], offset: ?>> loc(#loc33)
    %cast_19 = memref.cast %reinterpret_cast_18 : memref<7xi32, strided<[1], offset: ?>> to memref<7xi32> loc(#loc33)
    %c5_i32 = arith.constant 5 : i32 loc(#loc34)
    %reinterpret_cast_20 = memref.reinterpret_cast %cast_19 to offset: [0], sizes: [], strides: [] : memref<7xi32> to memref<i32> loc(#loc33)
    %5 = arith.index_cast %c5_i32 : i32 to index loc(#loc35)
    %reinterpret_cast_21 = memref.reinterpret_cast %reinterpret_cast_20 to offset: [%5], sizes: [], strides: [] : memref<i32> to memref<i32, strided<[], offset: ?>> loc(#loc35)
    %cast_22 = memref.cast %reinterpret_cast_21 : memref<i32, strided<[], offset: ?>> to memref<i32> loc(#loc35)
    %6 = memref.load %cast_22[] : memref<i32> loc(#loc33)
    memref.store %6, %alloca_6[] : memref<i32> loc(#loc57)
    %c0_i32_23 = arith.constant 0 : i32 loc(#loc36)
    %reinterpret_cast_24 = memref.reinterpret_cast %alloca_1 to offset: [0], sizes: [7], strides: [1] : memref<4x7xi32> to memref<7xi32> loc(#loc37)
    %7 = arith.index_cast %c0_i32_23 : i32 to index loc(#loc37)
    %reinterpret_cast_25 = memref.reinterpret_cast %reinterpret_cast_24 to offset: [%7], sizes: [7], strides: [1] : memref<7xi32> to memref<7xi32, strided<[1], offset: ?>> loc(#loc37)
    %cast_26 = memref.cast %reinterpret_cast_25 : memref<7xi32, strided<[1], offset: ?>> to memref<7xi32> loc(#loc37)
    %reinterpret_cast_27 = memref.reinterpret_cast %cast_26 to offset: [0], sizes: [], strides: [] : memref<7xi32> to memref<i32> loc(#loc64)
    memref.store %reinterpret_cast_27, %alloca_7[] : memref<memref<i32>> loc(#loc58)
    %c3_i32_28 = arith.constant 3 : i32 loc(#loc38)
    %reinterpret_cast_29 = memref.reinterpret_cast %alloca_1 to offset: [0], sizes: [7], strides: [1] : memref<4x7xi32> to memref<7xi32> loc(#loc39)
    %8 = arith.index_cast %c3_i32_28 : i32 to index loc(#loc39)
    %reinterpret_cast_30 = memref.reinterpret_cast %reinterpret_cast_29 to offset: [%8], sizes: [7], strides: [1] : memref<7xi32> to memref<7xi32, strided<[1], offset: ?>> loc(#loc39)
    %cast_31 = memref.cast %reinterpret_cast_30 : memref<7xi32, strided<[1], offset: ?>> to memref<7xi32> loc(#loc39)
    %c6_i32 = arith.constant 6 : i32 loc(#loc40)
    %reinterpret_cast_32 = memref.reinterpret_cast %cast_31 to offset: [0], sizes: [], strides: [] : memref<7xi32> to memref<i32> loc(#loc39)
    %9 = arith.index_cast %c6_i32 : i32 to index loc(#loc41)
    %reinterpret_cast_33 = memref.reinterpret_cast %reinterpret_cast_32 to offset: [%9], sizes: [], strides: [] : memref<i32> to memref<i32, strided<[], offset: ?>> loc(#loc41)
    %cast_34 = memref.cast %reinterpret_cast_33 : memref<i32, strided<[], offset: ?>> to memref<i32> loc(#loc41)
    memref.store %cast_34, %alloca_8[] : memref<memref<i32>> loc(#loc59)
    %c2_i32 = arith.constant 2 : i32 loc(#loc42)
    %reinterpret_cast_35 = memref.reinterpret_cast %alloca_1 to offset: [0], sizes: [7], strides: [1] : memref<4x7xi32> to memref<7xi32> loc(#loc43)
    %10 = arith.index_cast %c2_i32 : i32 to index loc(#loc43)
    %reinterpret_cast_36 = memref.reinterpret_cast %reinterpret_cast_35 to offset: [%10], sizes: [7], strides: [1] : memref<7xi32> to memref<7xi32, strided<[1], offset: ?>> loc(#loc43)
    %cast_37 = memref.cast %reinterpret_cast_36 : memref<7xi32, strided<[1], offset: ?>> to memref<7xi32> loc(#loc43)
    memref.store %cast_37, %alloca_9[] : memref<memref<7xi32>> loc(#loc60)
    %c2_i32_38 = arith.constant 2 : i32 loc(#loc44)
    %reinterpret_cast_39 = memref.reinterpret_cast %alloca_1 to offset: [0], sizes: [7], strides: [1] : memref<4x7xi32> to memref<7xi32> loc(#loc45)
    %11 = arith.index_cast %c2_i32_38 : i32 to index loc(#loc45)
    %reinterpret_cast_40 = memref.reinterpret_cast %reinterpret_cast_39 to offset: [%11], sizes: [7], strides: [1] : memref<7xi32> to memref<7xi32, strided<[1], offset: ?>> loc(#loc45)
    %cast_41 = memref.cast %reinterpret_cast_40 : memref<7xi32, strided<[1], offset: ?>> to memref<7xi32> loc(#loc45)
    %reinterpret_cast_42 = memref.reinterpret_cast %cast_41 to offset: [0], sizes: [], strides: [] : memref<7xi32> to memref<i32> loc(#loc65)
    memref.store %reinterpret_cast_42, %alloca_10[] : memref<memref<i32>> loc(#loc61)
    %c3_i32_43 = arith.constant 3 : i32 loc(#loc46)
    %reinterpret_cast_44 = memref.reinterpret_cast %alloca_0 to offset: [0], sizes: [], strides: [] : memref<10xi32> to memref<i32> loc(#loc47)
    %12 = arith.index_cast %c3_i32_43 : i32 to index loc(#loc47)
    %reinterpret_cast_45 = memref.reinterpret_cast %reinterpret_cast_44 to offset: [%12], sizes: [], strides: [] : memref<i32> to memref<i32, strided<[], offset: ?>> loc(#loc47)
    %cast_46 = memref.cast %reinterpret_cast_45 : memref<i32, strided<[], offset: ?>> to memref<i32> loc(#loc47)
    %13 = memref.load %cast_46[] : memref<i32> loc(#loc47)
    memref.store %13, %alloca[] : memref<i32> loc(#loc66)
    %14 = memref.load %alloca[] : memref<i32> loc(#loc66)
    return %14 : i32 loc(#loc66)
  } loc(#loc50)
} loc(#loc)
*/
