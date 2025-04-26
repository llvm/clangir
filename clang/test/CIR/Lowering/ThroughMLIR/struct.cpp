// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// Check the MLIR lowering of struct and member accesses
struct s {
  int a;
  double b;
  char c;
  float d[5];
};

int main() {
  s v;
  // CHECK: %[[ALLOCA:.+]] = memref.alloca() {alignment = 8 : i64} : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>>
  v.a = 7;
  // CHECK: %[[C_7:.+]] = arith.constant 7 : i32
  // CHECK: %[[I8_EQUIV_A:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>> to memref<40xi8>
  // CHECK: %[[OFFSET_A:.+]] = arith.constant 0 : index
  // CHECK: %[[VIEW_A:.+]] = memref.view %[[I8_EQUIV_A]][%[[OFFSET_A]]][] : memref<40xi8> to memref<i32>
  // CHECK: memref.store %[[C_7]], %[[VIEW_A]][] : memref<i32>

  v.b = 3.;
  // CHECK: %[[C_3:.+]] = arith.constant 3.000000e+00 : f64
  // CHECK: %[[I8_EQUIV_B:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>> to memref<40xi8>
  // CHECK: %[[OFFSET_B:.+]] = arith.constant 8 : index
  // CHECK: %[[VIEW_B:.+]] = memref.view %[[I8_EQUIV_B]][%[[OFFSET_B]]][] : memref<40xi8> to memref<f64>
  // CHECK: memref.store %[[C_3]], %[[VIEW_B]][] : memref<f64>

  v.c = 'z';
  // CHECK: %[[C_122:.+]] = arith.constant 122 : i8
  // CHECK: %[[I8_EQUIV_C:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>> to memref<40xi8>
  // CHECK: %[[OFFSET_C:.+]] = arith.constant 16 : index
  // CHECK: %[[VIEW_C:.+]] = memref.view %[[I8_EQUIV_C]][%[[OFFSET_C]]][] : memref<40xi8> to memref<i8>
  // memref.store %[[C_122]], %[[VIEW_C]][] : memref<i8>

  auto& a = v.d;
  v.d[4] = 6.f;
  // CHECK: %[[C_6:.+]] = arith.constant 6.000000e+00 : f32
  // CHECK: %[[I8_EQUIV_D:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>> to memref<40xi8>
  // CHECK: %[[OFFSET_D:.+]] = arith.constant 20 : index
  // CHECK: %[[VIEW_D:.+]] = memref.view %[[I8_EQUIV_D]][%[[OFFSET_D]]][] : memref<40xi8> to memref<5xf32>
  // CHECK: %[[C_4:.+]] = arith.constant 4 : i32
  // CHECK: %[[I_D:.+]] = arith.index_cast %[[C_4]] : i32 to index
  // CHECK: memref.store %[[C_6]], %[[VIEW_D]][%[[I_D]]] : memref<5xf32>

  return v.c;
  // CHECK: %[[I8_EQUIV_C_1:.+]] = named_tuple.cast %[[ALLOCA]] : memref<!named_tuple.named_tuple<"s", [i32, f64, i8, tensor<5xf32>]>> to memref<40xi8>
  // CHECK: %[[OFFSET_C_1:.+]] = arith.constant 16 : index
  // CHECK: %[[VIEW_C_1:.+]] = memref.view %[[I8_EQUIV_C_1]][%[[OFFSET_C_1]]][] : memref<40xi8> to memref<i8>
  // CHECK: %[[VALUE_C:.+]] = memref.load %[[VIEW_C_1]][] : memref<i8>
  // CHECK: %[[VALUE_RET:.+]] = arith.extsi %[[VALUE_C]] : i8 to i32
 }
