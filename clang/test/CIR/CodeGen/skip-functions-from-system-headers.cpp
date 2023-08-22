// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir -fclangir-skip-system-headers -I%S/../Inputs %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "skip-this-header.h"

void test() {
  String s1{};
  String s2{1};
  String s3{"abcdefghijklmnop"};
}

// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC2Ev
// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC2Ei
// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC2EPKc
// CHECK-NOT: cir.func linkonce_odr @_ZN6StringC1EPKc

// CHECK: cir.func @_Z4testv()
// CHECK:   cir.call @_ZN6StringC1Ev(%0) : (!cir.ptr<!ty_22String22>) -> ()