// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
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
// CHECK: !ty_22A22 = !cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}>
// CHECK: !ty_22B22 = !cir.struct<struct "B" packed {!cir.int<s, 32>, !cir.int<s, 8>, !cir.array<!cir.struct<struct "A" packed {!cir.int<s, 32>, !cir.int<s, 8>}> x 6>}>

// CHECK: cir.func {{.*@foo()}}
// CHECK:  %0 = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["a"] {alignment = 1 : i64}
// CHECK:  %1 = cir.alloca !ty_22B22, cir.ptr <!ty_22B22>, ["b"] {alignment = 1 : i64}
void foo() {
    A a;
    B b;
}


