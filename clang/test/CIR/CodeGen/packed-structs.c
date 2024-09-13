// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#pragma pack(1)

typedef struct {
    int  a0;
    char a1;    
} A;

typedef struct {
    int  b0;
    char b1;
    A a[6];    
} B;

typedef struct {
    int  c0;
    char c1;    
} __attribute__((aligned(2))) C;


// CHECK: !ty_A = !cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}>
// CHECK: !ty_C = !cir.struct<struct "C" packed {!cir.int<s, 32>, !cir.int<s, 8>, !cir.int<u, 8>}>
// CHECK: !ty_D = !cir.struct<struct "D" packed {!cir.int<s, 8>, !cir.int<u, 8>, !cir.int<s, 32>}
// CHECK: !ty_F = !cir.struct<struct "F" packed {!cir.int<s, 64>, !cir.int<s, 8>}
// CHECK: !ty_E = !cir.struct<struct "E" packed {!cir.struct<struct "D" packed {!cir.int<s, 8>, !cir.int<u, 8>, !cir.int<s, 32>}
// CHECK: !ty_G = !cir.struct<struct "G" {!cir.struct<struct "F" packed {!cir.int<s, 64>, !cir.int<s, 8>}
// CHECK: !ty_H = !cir.struct<struct "H" {!cir.int<s, 32>, !cir.struct<union "anon.1" {!cir.int<s, 8>, !cir.int<s, 32>}
// CHECK: !ty_B = !cir.struct<struct "B" packed {!cir.int<s, 32>, !cir.int<s, 8>, !cir.array<!cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}> x 6>}>
// CHECK: !ty_I = !cir.struct<struct "I" packed {!cir.int<s, 8>, !cir.struct<struct "H" {!cir.int<s, 32>, !cir.struct<union "anon.1" {!cir.int<s, 8>, !cir.int<s, 32>}
// CHECK: !ty_J = !cir.struct<struct "J" packed {!cir.int<s, 8>, !cir.int<s, 8>, !cir.int<s, 8>, !cir.int<s, 8>, !cir.struct<struct "I" packed {!cir.int<s, 8>, !cir.struct<struct "H" {!cir.int<s, 32>, !cir.struct<union "anon.1" {!cir.int<s, 8>, !cir.int<s, 32>}

// CHECK: cir.func {{.*@foo()}}
// CHECK:  %0 = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["a"] {alignment = 1 : i64}
// CHECK:  %1 = cir.alloca !ty_B, !cir.ptr<!ty_B>, ["b"] {alignment = 1 : i64}
// CHECK:  %2 = cir.alloca !ty_C, !cir.ptr<!ty_C>, ["c"] {alignment = 2 : i64}
void foo() {
    A a;
    B b;
    C c;
}

#pragma pack(2)

typedef struct {
    char b;
    int c;
} D;

typedef struct {
    D e;
    int f;
} E;

// CHECK: cir.func {{.*@f1()}}
// CHECK:  %0 = cir.alloca !ty_E, !cir.ptr<!ty_E>, ["a"] {alignment = 2 : i64}
void f1() {
    E a = {};
}

#pragma pack(1)

typedef struct {
    long b;
    char c;
} F;

typedef struct {
    F e;
    char f;
} G;

// CHECK: cir.func {{.*@f2()}}
// CHECK:  %0 = cir.alloca !ty_G, !cir.ptr<!ty_G>, ["a"] {alignment = 1 : i64}
void f2() {
    G a = {};
}

#pragma pack(1)

typedef struct {
    int d0;
    union {
        char null;
        int val;
    } value;
} H;

typedef struct {
    char t;
    H d;
} I;

typedef struct {
    char a0;
    char a1;
    char a2;
    char a3;
    I c;
    int a;
} J;

// CHECK: cir.func {{.*@f3()}}
// CHECK:  %0 = cir.alloca !ty_J, !cir.ptr<!ty_J>, ["a"] {alignment = 1 : i64}
void f3() {
    J a = {0};
}