// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Struk {
  int a;
  Struk() {}
  void test() {}
};

void baz() {
  Struk s;
}

// CHECK: !22struct2EStruk22 = !cir.struct<"struct.Struk", i32>

// CHECK:   cir.func @_ZN5StrukC2Ev(%arg0: !cir.ptr<!22struct2EStruk22>
// CHECK-NEXT:     %0 = cir.alloca !cir.ptr<!22struct2EStruk22>, cir.ptr <!cir.ptr<!22struct2EStruk22>>, ["this", paraminit] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %0 : !cir.ptr<!22struct2EStruk22>, cir.ptr <!cir.ptr<!22struct2EStruk22>>
// CHECK-NEXT:     %1 = cir.load %0 : cir.ptr <!cir.ptr<!22struct2EStruk22>>, !cir.ptr<!22struct2EStruk22>
// CHECK-NEXT:     cir.return

// CHECK:   cir.func @_ZN5StrukC1Ev(%arg0: !cir.ptr<!22struct2EStruk22>
// CHECK-NEXT:     %0 = cir.alloca !cir.ptr<!22struct2EStruk22>, cir.ptr <!cir.ptr<!22struct2EStruk22>>, ["this", paraminit] {alignment = 8 : i64}
// CHECK-NEXT:     cir.store %arg0, %0 : !cir.ptr<!22struct2EStruk22>, cir.ptr <!cir.ptr<!22struct2EStruk22>>
// CHECK-NEXT:     %1 = cir.load %0 : cir.ptr <!cir.ptr<!22struct2EStruk22>>, !cir.ptr<!22struct2EStruk22>
// CHECK-NEXT:     cir.call @_ZN5StrukC2Ev(%1) : (!cir.ptr<!22struct2EStruk22>) -> ()
// CHECK-NEXT:     cir.return

// CHECK:   cir.func @_Z3bazv()
// CHECK-NEXT:     %0 = cir.alloca !22struct2EStruk22, cir.ptr <!22struct2EStruk22>, ["s", uninitialized] {alignment = 4 : i64}
// CHECK-NEXT:     cir.call @_ZN5StrukC1Ev(%0) : (!cir.ptr<!22struct2EStruk22>) -> ()
// CHECK-NEXT:     cir.return
