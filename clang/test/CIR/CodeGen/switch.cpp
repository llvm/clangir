// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void sw1(int a) {
  switch (int b = 1; a) {
  case 0:
    b = b + 1;
    break;
  case 1:
    break;
  case 2: {
    b = b + 1;
    int yolo = 100;
    break;
  }
  }
}

// CHECK: func @_Z3sw1i
// CHECK: cir.switch (%3 : i32) [
// CHECK-NEXT: case (equal, 0 : i32)  {
// CHECK-NEXT:   %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield break
// CHECK-NEXT: },
// CHECK-NEXT: case (equal, 1 : i32)  {
// CHECK-NEXT:   cir.yield break
// CHECK-NEXT: },
// CHECK-NEXT: case (equal, 2 : i32)  {
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:       %4 = cir.alloca i32, cir.ptr <i32>, ["yolo", cinit]
// CHECK-NEXT:       %5 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %6 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %7 = cir.binop(add, %5, %6) : i32
// CHECK-NEXT:       cir.store %7, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:       %8 = cir.cst(100 : i32) : i32
// CHECK-NEXT:       cir.store %8, %4 : i32, cir.ptr <i32>
// CHECK-NEXT:       cir.yield break
// CHECK-NEXT:     }
// CHECK-NEXT:     cir.yield fallthrough
// CHECK-NEXT:   }

void sw2(int a) {
  switch (int yolo = 2; a) {
  case 3:
    // "fomo" has the same lifetime as "yolo"
    int fomo = 0;
    yolo = yolo + fomo;
    break;
  }
}

// CHECK: func @_Z3sw2i
// CHECK: cir.scope {
// CHECK-NEXT:   %1 = cir.alloca i32, cir.ptr <i32>, ["yolo", cinit]
// CHECK-NEXT:   %2 = cir.alloca i32, cir.ptr <i32>, ["fomo", cinit]
// CHECK:        cir.switch (%4 : i32) [
// CHECK-NEXT:   case (equal, 3 : i32)  {
// CHECK-NEXT:     %5 = cir.cst(0 : i32) : i32
// CHECK-NEXT:     cir.store %5, %2 : i32, cir.ptr <i32>

void sw3(int a) {
  switch (a) {
  default:
    break;
  }
}

// CHECK: func @_Z3sw3i
// CHECK: cir.scope {
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.switch (%1 : i32) [
// CHECK-NEXT:   case (default)  {
// CHECK-NEXT:     cir.yield break
// CHECK-NEXT:   }
// CHECK-NEXT:   ]

int sw4(int a) {
  switch (a) {
  case 42: {
    return 3;
  }
  default:
    return 2;
  }
  return 0;
}

// CHECK: func @_Z3sw4i
// CHECK:       cir.switch (%4 : i32) [
// CHECK-NEXT:       case (equal, 42 : i32)  {
// CHECK-NEXT:         cir.scope {
// CHECK-NEXT:           %5 = cir.cst(3 : i32) : i32
// CHECK-NEXT:           cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:           %6 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:           cir.return %6 : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         cir.yield fallthrough
// CHECK-NEXT:       },
// CHECK-NEXT:       case (default)  {
// CHECK-NEXT:         %5 = cir.cst(2 : i32) : i32
// CHECK-NEXT:         cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:         %6 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:         cir.return %6 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       ]

void sw5(int a) {
  switch (a) {
  case 1:;
  }
}

// CHECK: func @_Z3sw5i
// CHECK: cir.switch (%1 : i32) [
// CHECK-NEXT:   case (equal, 1 : i32)  {
// CHECK-NEXT:     cir.yield fallthrough

void sw6(int a) {
  switch (a) {
  case 0:
  case 1:
  case 2:
    break;
  case 3:
  case 4:
  case 5:
    break;
  }
}

// CHECK: func @_Z3sw6i
// CHECK: cir.switch (%1 : i32) [
// CHECK-NEXT: case (anyof, [0, 1, 2] : i32)  {
// CHECK-NEXT:   cir.yield break
// CHECK-NEXT: },
// CHECK-NEXT: case (anyof, [3, 4, 5] : i32)  {
// CHECK-NEXT:   cir.yield break
// CHECK-NEXT: }

void sw7(int a) {
  switch (a) {
  case 0:
  case 1:
  case 2:
    int x;
  case 3:
  case 4:
  case 5:
    break;
  }
}

// CHECK: func @_Z3sw7i
// CHECK: case (anyof, [0, 1, 2] : i32)  {
// CHECK-NEXT:   cir.yield fallthrough
// CHECK-NEXT: },
// CHECK-NEXT: case (anyof, [3, 4, 5] : i32)  {
// CHECK-NEXT:   cir.yield break
// CHECK-NEXT: }
