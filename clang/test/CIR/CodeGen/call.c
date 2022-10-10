// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o - | FileCheck %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o - | FileCheck %s --check-prefix=CXX

void a(void) {}
int b(int a, int b) {
  return a + b;
}
double c(double a, double b) {
  return a + b;
}

void d(void) {
  a();
  b(0, 1);
}

// CHECK: module  {
// CHECK:   func @a() {
// CHECK:     cir.return
// CHECK:   }
// CHECK:   func @b(%arg0: i32 {{.*}}, %arg1: i32 {{.*}}) -> i32 {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["a", paraminit]
// CHECK:     %1 = cir.alloca i32, cir.ptr <i32>, ["b", paraminit]
// CHECK:     %2 = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized]
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK:     cir.store %arg1, %1 : i32, cir.ptr <i32>
// CHECK:     %3 = cir.load %0 : cir.ptr <i32>, i32
// CHECK:     %4 = cir.load %1 : cir.ptr <i32>, i32
// CHECK:     %5 = cir.binop(add, %3, %4) : i32
// CHECK:     cir.store %5, %2 : i32, cir.ptr <i32>
// CHECK:     %6 = cir.load %2 : cir.ptr <i32>, i32
// CHECK:     cir.return %6
// CHECK:   }
// CHECK:   func @c(%arg0: f64 {{.*}}, %arg1: f64 {{.*}}) -> f64 {
// CHECK:     %0 = cir.alloca f64, cir.ptr <f64>, ["a", paraminit]
// CHECK:     %1 = cir.alloca f64, cir.ptr <f64>, ["b", paraminit]
// CHECK:     %2 = cir.alloca f64, cir.ptr <f64>, ["__retval", uninitialized]
// CHECK:     cir.store %arg0, %0 : f64, cir.ptr <f64>
// CHECK:     cir.store %arg1, %1 : f64, cir.ptr <f64>
// CHECK:     %3 = cir.load %0 : cir.ptr <f64>, f64
// CHECK:     %4 = cir.load %1 : cir.ptr <f64>, f64
// CHECK:     %5 = cir.binop(add, %3, %4) : f64
// CHECK:     cir.store %5, %2 : f64, cir.ptr <f64>
// CHECK:     %6 = cir.load %2 : cir.ptr <f64>, f64
// CHECK:     cir.return %6 : f64
// CHECK:   }
// CHECK:   func @d() {
// CHECK:     call @a() : () -> ()
// CHECK:     %0 = cir.cst(0 : i32) : i32
// CHECK:     %1 = cir.cst(1 : i32) : i32
// CHECK:     call @b(%0, %1) : (i32, i32) -> i32
// CHECK:     cir.return
// CHECK:   }
//
// CXX: module  {
// CXX-NEXT:   func @_Z1av() {
// CXX-NEXT:     cir.return
// CXX-NEXT:   }
// CXX-NEXT:   func @_Z1bii(%arg0: i32 {{.*}}, %arg1: i32 {{.*}}) -> i32 {
// CXX-NEXT:     %0 = cir.alloca i32, cir.ptr <i32>, ["a", paraminit]
// CXX-NEXT:     %1 = cir.alloca i32, cir.ptr <i32>, ["b", paraminit]
// CXX-NEXT:     %2 = cir.alloca i32, cir.ptr <i32>, ["__retval", uninitialized]
// CXX-NEXT:     cir.store %arg0, %0 : i32, cir.ptr <i32>
// CXX-NEXT:     cir.store %arg1, %1 : i32, cir.ptr <i32>
// CXX-NEXT:     %3 = cir.load %0 : cir.ptr <i32>, i32
// CXX-NEXT:     %4 = cir.load %1 : cir.ptr <i32>, i32
// CXX-NEXT:     %5 = cir.binop(add, %3, %4) : i32
// CXX-NEXT:     cir.store %5, %2 : i32, cir.ptr <i32>
// CXX-NEXT:     %6 = cir.load %2 : cir.ptr <i32>, i32
// CXX-NEXT:     cir.return %6
// CXX-NEXT:   }
// CXX-NEXT:   func @_Z1cdd(%arg0: f64 {{.*}}, %arg1: f64 {{.*}}) -> f64 {
// CXX-NEXT:     %0 = cir.alloca f64, cir.ptr <f64>, ["a", paraminit]
// CXX-NEXT:     %1 = cir.alloca f64, cir.ptr <f64>, ["b", paraminit]
// CXX-NEXT:     %2 = cir.alloca f64, cir.ptr <f64>, ["__retval", uninitialized]
// CXX-NEXT:     cir.store %arg0, %0 : f64, cir.ptr <f64>
// CXX-NEXT:     cir.store %arg1, %1 : f64, cir.ptr <f64>
// CXX-NEXT:     %3 = cir.load %0 : cir.ptr <f64>, f64
// CXX-NEXT:     %4 = cir.load %1 : cir.ptr <f64>, f64
// CXX-NEXT:     %5 = cir.binop(add, %3, %4) : f64
// CXX-NEXT:     cir.store %5, %2 : f64, cir.ptr <f64>
// CXX-NEXT:     %6 = cir.load %2 : cir.ptr <f64>, f64
// CXX-NEXT:     cir.return %6 : f64
// CXX-NEXT:   }
// CXX-NEXT:   func @_Z1dv() {
// CXX-NEXT:     call @_Z1av() : () -> ()
// CXX-NEXT:     %0 = cir.cst(0 : i32) : i32
// CXX-NEXT:     %1 = cir.cst(1 : i32) : i32
// CXX-NEXT:     call @_Z1bii(%0, %1) : (i32, i32) -> i32
// CXX-NEXT:     cir.return
// CXX-NEXT:   }
