// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

int sum() {
  int s = 0;
  int i = 0;
  do {
    s += i;
    ++i;
  } while (i <= 10);
  return s;
}

// CHECK: func.func @sum() -> i32 {
// CHECK: %[[ALLOC:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK: %[[ALLOC0:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK: %[[ALLOC1:.+]] = memref.alloca() {alignment = 4 : i64} : memref<i32>
// CHECK: %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK: memref.store %[[C0_I32]], %[[ALLOC0]][] : memref<i32>
// CHECK: %[[C0_I32_2:.+]] = arith.constant 0 : i32
// CHECK: memref.store %[[C0_I32_2]], %[[ALLOC1]][] : memref<i32>
// CHECK: memref.alloca_scope {
// CHECK:   scf.while : () -> () {
// CHECK:     %[[VAR1:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
// CHECK:     %[[VAR2:.+]] = memref.load %[[ALLOC0]][] : memref<i32>
// CHECK:     %[[ADD:.+]] = arith.addi %[[VAR2]], %[[VAR1]] : i32
// CHECK:     memref.store %[[ADD]], %[[ALLOC0]][] : memref<i32>
// CHECK:     %[[VAR3:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
// CHECK:     %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK:     %[[ADD1:.+]] = arith.addi %[[VAR3]], %[[C1_I32]] : i32
// CHECK:     memref.store %[[ADD1]], %[[ALLOC1]][] : memref<i32>
// CHECK:     %[[VAR4:.+]] = memref.load %[[ALLOC1]][] : memref<i32>
// CHECK:     %[[C10_I32:.+]] = arith.constant 10 : i32
// CHECK:     %[[CMP:.+]] = arith.cmpi sle, %[[VAR4]], %[[C10_I32]] : i32
// CHECK:     %[[EXT:.+]] = arith.extui %[[CMP]] : i1 to i32
// CHECK:     %[[C0_I32_3:.+]] = arith.constant 0 : i32
// CHECK:     %[[NE:.+]] = arith.cmpi ne, %[[EXT]], %[[C0_I32_3]] : i32
// CHECK:     %[[EXT1:.+]] = arith.extui %[[NE]] : i1 to i8
// CHECK:     %[[TRUNC:.+]] = arith.trunci %[[EXT1]] : i8 to i1
// CHECK:     scf.condition(%[[TRUNC]])
// CHECK:   } do {
// CHECK:     scf.yield
// CHECK:   }
// CHECK: }
// CHECK: %[[LOAD:.+]] = memref.load %[[ALLOC0]][] : memref<i32>
// CHECK: memref.store %[[LOAD]], %[[ALLOC]][] : memref<i32>
// CHECK: %[[RET:.+]] = memref.load %[[ALLOC]][] : memref<i32>
// CHECK: return %[[RET]] : i32
