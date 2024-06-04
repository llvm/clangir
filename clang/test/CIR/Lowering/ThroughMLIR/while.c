// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void foo() {
  int a = 0;
  while(a < 2) {
    a++;
  }
}

//CHECK: func.func @foo() {
//CHECK:   %alloca = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %c0_i32 = arith.constant 0 : i32 
//CHECK:   memref.store %c0_i32, %alloca[] : memref<i32> 
//CHECK:   memref.alloca_scope  {
//CHECK:     scf.while : () -> () {
//CHECK:       %0 = memref.load %alloca[] : memref<i32> 
//CHECK:       %c2_i32 = arith.constant 2 : i32 
//CHECK:       %1 = arith.cmpi ult, %0, %c2_i32 : i32 
//CHECK:       %2 = arith.extui %1 : i1 to i32 
//CHECK:       %c0_i32_0 = arith.constant 0 : i32 
//CHECK:       %3 = arith.cmpi ne, %2, %c0_i32_0 : i32 
//CHECK:       %4 = arith.extui %3 : i1 to i8 
//CHECK:       %5 = arith.trunci %4 : i8 to i1 
//CHECK:       scf.condition(%5) 
//CHECK:     } do {
//CHECK:       %0 = memref.load %alloca[] : memref<i32> 
//CHECK:       %c1_i32 = arith.constant 1 : i32 
//CHECK:       %1 = arith.addi %0, %c1_i32 : i32 
//CHECK:       memref.store %1, %alloca[] : memref<i32> 
//CHECK:       scf.yield 
//CHECK:     } 
//CHECK:  } 
//CHECK:   return 
//CHECK: } 