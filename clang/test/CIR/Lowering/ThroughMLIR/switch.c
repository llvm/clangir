// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

int switch_test(int cond) {

  // CHECK:   %alloca = memref.alloca() {alignment = 4 : i64} : memref<i32>
  // CHECK:   %alloca_0 = memref.alloca() {alignment = 4 : i64} : memref<i32>
  // CHECK:   %alloca_1 = memref.alloca() {alignment = 4 : i64} : memref<i32>

  // CHECK:   memref.store %arg0, %alloca[] : memref<i32>

  int ret;

  // CHECK:   memref.alloca_scope  {

  // CHECK:     %2 = memref.load %alloca[] : memref<i32>
  // CHECK:     %3 = arith.index_cast %2 : i32 to index

  switch (cond) {

  // CHECK:     scf.index_switch %3 

    case 0: ret = 10; break;

    // CHECK:     case 0 {
    // CHECK:       %c100_i32 = arith.constant 100 : i32
    // CHECK:       memref.store %c100_i32, %alloca_1[] : memref<i32>
    // CHECK:       scf.yield
    // CHECK:     }

    case 1: ret = 100; break;

    // CHECK:     case 1 {
    // CHECK:       %c1000_i32 = arith.constant 1000 : i32
    // CHECK:       memref.store %c1000_i32, %alloca_1[] : memref<i32>
    // CHECK:       scf.yield
    // CHECK:     }

    default: ret = 1000; break;

    // CHECK:     default {
    // CHECK:       %c10_i32 = arith.constant 10 : i32
    // CHECK:       memref.store %c10_i32, %alloca_1[] : memref<i32>
    // CHECK:     }

  }

  return ret;

  // CHECK:   }

  // CHECK:   %0 = memref.load %alloca_1[] : memref<i32>
  // CHECK:   memref.store %0, %alloca_0[] : memref<i32>
  // CHECK:   %1 = memref.load %alloca_0[] : memref<i32>
  // CHECK:   return %1 : i32
}
