// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// This case is not yet implemented.
// XFAIL: *

struct Other {
    int x;
};

struct Trivial {
    int x;
    double y;
    decltype(&Other::x) ptr;
};

// This case has a trivial default constructor, but can't be zero-initialized.
Trivial t;

// The type is wrong on the last member here. It needs to be initialized to -1,
//   but I don't know what that will look like.
// CHECK: cir.global {{.*}} @t = #cir.const_record<{#cir.int<0> : !s32i, #cir.fp<0.000000e+00> : !cir.double, #cir.int<-1> : !s64i}>
