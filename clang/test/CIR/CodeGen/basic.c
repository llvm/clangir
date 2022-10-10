// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -fcir-warnings %s -fcir-output=%t.cir
// RUN: FileCheck --input-file=%t.cir %s

int foo(int i) {
  return i;
}

// CHECK: module  {
// CHECK-NEXT: func @foo(%arg0: i32) -> i32 {
// CHECK-NEXT:   %0 = cir.alloca i32 = uninitialized, cir.ptr <i32>
// CHECK-NEXT:   cir.store %arg0, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:   %1 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:   cir.return %1 : i32
// CHECK-NEXT: }
