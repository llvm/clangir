// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o - | FileCheck %s

struct String {
  long size;
};

void split(String &S) {}

// CHECK: cir.func @_Z5splitR6String(%arg0: !cir.ptr<!ty_22String22>
// CHECK:     %0 = cir.alloca !cir.ptr<!ty_22String22>, cir.ptr <!cir.ptr<!ty_22String22>>, ["S", init]

void foo() {
  String s;
  split(s);
}

// CHECK: cir.func @_Z3foov()
// CHECK:     %0 = cir.alloca !ty_22String22, cir.ptr <!ty_22String22>, ["s"]
// CHECK:     cir.call @_Z5splitR6String(%0) : (!cir.ptr<!ty_22String22>) -> ()
