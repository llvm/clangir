// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Bar {
  int a;
  char b;
};

struct Foo {
  int a;
  char b;
  Bar z;
};

void baz() {
  Bar b;
  Foo f;
}

//      CHECK: !22struct2EBar22 = !cir.struct<"struct.Bar", i32, i8>
// CHECK-NEXT: !22struct2EFoo22 = !cir.struct<"struct.Foo", i32, i8, !22struct2EBar22>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func @baz() {
// CHECK-NEXT:     %0 = cir.alloca !22struct2EFoo22, cir.ptr <!22struct2EFoo22>, [uninitialized] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.alloca !22struct2EBar22, cir.ptr <!22struct2EBar22>, [uninitialized] {alignment = 4 : i64}
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
