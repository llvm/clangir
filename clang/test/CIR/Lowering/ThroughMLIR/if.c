// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

void foo() {
  int a = 2;
  int b = 0;
  if (a > 0) {
    b++;
  } else {
    b--;
  }
}

//CHECK: func.func @foo() {
//CHECK:   %alloca = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %alloca_0 = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:   %c2_i32 = arith.constant 2 : i32 
//CHECK:   memref.store %c2_i32, %alloca[] : memref<i32> 
//CHECK:   %c0_i32 = arith.constant 0 : i32 
//CHECK:   memref.store %c0_i32, %alloca_0[] : memref<i32> 
//CHECK:   memref.alloca_scope  {
//CHECK:     %0 = memref.load %alloca[] : memref<i32> 
//CHECK:     %c0_i32_1 = arith.constant 0 : i32 
//CHECK:     %1 = arith.cmpi ugt, %0, %c0_i32_1 : i32 
//CHECK:     %2 = arith.extui %1 : i1 to i32 
//CHECK:     %c0_i32_2 = arith.constant 0 : i32 
//CHECK:     %3 = arith.cmpi ne, %2, %c0_i32_2 : i32 
//CHECK:     %4 = arith.extui %3 : i1 to i8 
//CHECK:     %5 = arith.trunci %4 : i8 to i1 
//CHECK:     scf.if %5 {
//CHECK:       %6 = memref.load %alloca_0[] : memref<i32> 
//CHECK:       %c1_i32 = arith.constant 1 : i32 
//CHECK:       %7 = arith.addi %6, %c1_i32 : i32 
//CHECK:       memref.store %7, %alloca_0[] : memref<i32> 
//CHECK:     } else {
//CHECK:       %6 = memref.load %alloca_0[] : memref<i32> 
//CHECK:       %c1_i32 = arith.constant 1 : i32 
//CHECK:       %7 = arith.subi %6, %c1_i32 : i32 
//CHECK:       memref.store %7, %alloca_0[] : memref<i32> 
//CHECK:     } 
//CHECK:   } 
//CHECK:   return 
//CHECK: } 

void foo2() {
  int a = 2;
  int b = 0;
  if (a < 3) {
    b++;
  }
}

//CHECK: func.func @foo2() {
//CHECK:   %alloca = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:   %alloca_0 = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:   %c2_i32 = arith.constant 2 : i32 
//CHECK:   memref.store %c2_i32, %alloca[] : memref<i32> 
//CHECK:   %c0_i32 = arith.constant 0 : i32 
//CHECK:   memref.store %c0_i32, %alloca_0[] : memref<i32> 
//CHECK:   memref.alloca_scope  {
//CHECK:     %0 = memref.load %alloca[] : memref<i32> 
//CHECK:     %c3_i32 = arith.constant 3 : i32 
//CHECK:     %1 = arith.cmpi ult, %0, %c3_i32 : i32 
//CHECK:     %2 = arith.extui %1 : i1 to i32 
//CHECK:     %c0_i32_1 = arith.constant 0 : i32 
//CHECK:     %3 = arith.cmpi ne, %2, %c0_i32_1 : i32 
//CHECK:     %4 = arith.extui %3 : i1 to i8 
//CHECK:     %5 = arith.trunci %4 : i8 to i1 
//CHECK:     scf.if %5 {
//CHECK:       %6 = memref.load %alloca_0[] : memref<i32> 
//CHECK:       %c1_i32 = arith.constant 1 : i32 
//CHECK:       %7 = arith.addi %6, %c1_i32 : i32 
//CHECK:       memref.store %7, %alloca_0[] : memref<i32> 
//CHECK:     } 
//CHECK:   } 
//CHECK:   return 
//CHECK: } 

void foo3() {
  int a = 2;
  int b = 0;
  if (a < 3) {
    int c = 1;
    if (c > 2) {
      b++;
    } else {
      b--;
    }
  }
}


//CHECK: func.func @foo3() {
//CHECK:   %alloca = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %alloca_0 = memref.alloca() {alignment = 4 : i64} : memref<i32>
//CHECK:   %c2_i32 = arith.constant 2 : i32 
//CHECK:   memref.store %c2_i32, %alloca[] : memref<i32> 
//CHECK:   %c0_i32 = arith.constant 0 : i32 
//CHECK:   memref.store %c0_i32, %alloca_0[] : memref<i32> 
//CHECK:   memref.alloca_scope  {
//CHECK:     %0 = memref.load %alloca[] : memref<i32> 
//CHECK:     %c3_i32 = arith.constant 3 : i32 
//CHECK:     %1 = arith.cmpi ult, %0, %c3_i32 : i32 
//CHECK:     %2 = arith.extui %1 : i1 to i32 
//CHECK:     %c0_i32_1 = arith.constant 0 : i32 
//CHECK:     %3 = arith.cmpi ne, %2, %c0_i32_1 : i32 
//CHECK:     %4 = arith.extui %3 : i1 to i8 
//CHECK:     %5 = arith.trunci %4 : i8 to i1 
//CHECK:     scf.if %5 {
//CHECK:       %alloca_2 = memref.alloca() {alignment = 4 : i64} : memref<i32> 
//CHECK:       %c1_i32 = arith.constant 1 : i32 
//CHECK:       memref.store %c1_i32, %alloca_2[] : memref<i32> 
//CHECK:       memref.alloca_scope  {
//CHECK:         %6 = memref.load %alloca_2[] : memref<i32> 
//CHECK:         %c2_i32_3 = arith.constant 2 : i32 
//CHECK:         %7 = arith.cmpi ugt, %6, %c2_i32_3 : i32 
//CHECK:         %8 = arith.extui %7 : i1 to i32 
//CHECK:         %c0_i32_4 = arith.constant 0 : i32 
//CHECK:         %9 = arith.cmpi ne, %8, %c0_i32_4 : i32 
//CHECK:         %10 = arith.extui %9 : i1 to i8 
//CHECK:         %11 = arith.trunci %10 : i8 to i1 
//CHECK:         scf.if %11 {
//CHECK:           %12 = memref.load %alloca_0[] : memref<i32> 
//CHECK:           %c1_i32_5 = arith.constant 1 : i32 
//CHECK:           %13 = arith.addi %12, %c1_i32_5 : i32 
//CHECK:           memref.store %13, %alloca_0[] : memref<i32> 
//CHECK:         } else {
//CHECK:           %12 = memref.load %alloca_0[] : memref<i32> 
//CHECK:           %c1_i32_5 = arith.constant 1 : i32 
//CHECK:           %13 = arith.subi %12, %c1_i32_5 : i32 
//CHECK:           memref.store %13, %alloca_0[] : memref<i32> 
//CHECK:         } 
//CHECK:       } 
//CHECK:     } 
//CHECK:   } 
//CHECK:   return 
//CHECK: } 
