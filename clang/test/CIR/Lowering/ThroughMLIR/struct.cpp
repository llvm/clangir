// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

struct s {
  int a;
  double b;
  char c;
};

int main() {
  s v;
  // CHECK: %[[ALLOCA:.+]] = memref.alloca() {alignment = 8 : i64} : memref<!named_tuple.named_tuple<"s", [i32, f64, i8]>>
  v.a = 7;
  // CHECK: %[[C_7:.+]] = arith.constant 7 : i32
  // CHECK: %[[I8_EQUIV_A:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8]>> to memref<24xi8>
  // CHECK: %[[OFFSET_A:.+]] = arith.constant 0 : index
  // CHECK: %[[VIEW_A:.+]] = memref.view %[[I8_EQUIV_A]][%[[OFFSET_A]]][] : memref<24xi8> to memref<i32>
  // CHECK: memref.store %[[C_7]], %[[VIEW_A]][] : memref<i32>

  v.b = 3.;
  // CHECK: %[[C_3:.+]] = arith.constant 3.000000e+00 : f64
  // CHECK: %[[I8_EQUIV_B:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8]>> to memref<24xi8>
  // CHECK: %[[OFFSET_B:.+]] = arith.constant 8 : index
  // CHECK: %[[VIEW_B:.+]] = memref.view %[[I8_EQUIV_B]][%[[OFFSET_B]]][] : memref<24xi8> to memref<f64>
  // CHECK: memref.store %[[C_3]], %[[VIEW_B]][] : memref<f64>

  v.c = 'z';
  // CHECK: %[[C_122:.+]]  = arith.constant 122 : i8
  // CHECK: %[[I8_EQUIV_C:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8]>> to memref<24xi8>
  // CHECK: %[[OFFSET_C:.+]] = arith.constant 16 : index
  // CHECK: %[[VIEW_C:.+]] = memref.view %[[I8_EQUIV_C]][%[[OFFSET_C]]][] : memref<24xi8> to memref<i8>
  // memref.store %[[C_122]], %[[VIEW_C]][] : memref<i8>

  return v.c;
  // CHECK: %[[I8_EQUIV_C_1:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8]>> to memref<24xi8>
  // CHECK: %[[OFFSET_C_1:.+]] = arith.constant 16 : index
  // CHECK: %[[VIEW_C_1:.+]] = memref.view %[[I8_EQUIV_C_1]][%[[OFFSET_C_1]]][] : memref<24xi8> to memref<i8>
  // CHECK: %[[VALUE_C:.+]] = memref.load %[[VIEW_C_1]][] : memref<i8>
  // CHECK: %[[VALUE_RET:.+]] = arith.extsi %[[VALUE_C]] : i8 to i32
 }
