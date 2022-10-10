// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void a(void) {}
int b(int a, int b) {
  return a + b;
}

void c(void) {
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
// CHECK:   func @c() {
// CHECK:     call @a() : () -> ()
// CHECK:     %0 = cir.cst(0 : i32) : i32
// CHECK:     %1 = cir.cst(1 : i32) : i32
// CHECK:     call @b(%0, %1) : (i32, i32) -> i32
// CHECK:     cir.return
// CHECK:   }
