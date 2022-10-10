// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fenable-clangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int foo(int i);

int foo(int i) {
  i;
  return i;
}

// CHECK: module  {
// CHECK-NEXT: func @foo(%arg0: i32 loc({{.*}})) -> i32 {
// CHECK-NEXT: %0 = cir.alloca i32, cir.ptr <i32>, ["i", paraminit] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized] {alignment = 4 : i64}
// CHECK-NEXT: cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK-NEXT: %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT: %3 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT: cir.store %3, %1 : i32, cir.ptr <i32>
// CHECK-NEXT: %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT: cir.return %4 : i32

int f2() { return 3; }

// CHECK: func @f2() -> i32 {
// CHECK-NEXT: %0 = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.cst(3 : i32) : i32
// CHECK-NEXT: cir.store %1, %0 : i32, cir.ptr <i32>
// CHECK-NEXT: %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT: cir.return %2 : i32

int f3() {
  int i = 3;
  return i;
}

// CHECK: func @f3() -> i32 {
// CHECK-NEXT: %0 = cir.alloca i32, cir.ptr <i32>, ["i", cinit] {alignment = 4 : i64}
// CHECK-NEXT: %1 = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized] {alignment = 4 : i64}
// CHECK-NEXT: %2 = cir.cst(3 : i32) : i32
// CHECK-NEXT: cir.store %2, %0 : i32, cir.ptr <i32>
// CHECK-NEXT: %3 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT: cir.store %3, %1 : i32, cir.ptr <i32>
// CHECK-NEXT: %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT: cir.return %4 : i32
