// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct demo{
    int memberOne;
} demo;

void testOne() {    
    demo one = {1};
}

void testTwo() {
    typedef struct demo {
        int memberOne;
        int memberTwo;
    } demo;

    demo two = {1, 2};
}

// CHECK: !ty_22demo22 = !cir.struct<struct "demo" {!cir.int<s, 32>}>
// CHECK: !ty_22demo2E122 = !cir.struct<struct "demo.1" {!cir.int<s, 32>, !cir.int<s, 32>}>
// CHECK: cir.func no_proto @testOne() extra(#fn_attr) {
// CHECK: %0 = cir.alloca !ty_22demo22, !cir.ptr<!ty_22demo22>, ["one"] {alignment = 4 : i64}
// CHECK: %1 = cir.const #cir.const_struct<{#cir.int<1> : !s32i}> : !ty_22demo22
// CHECK: cir.store %1, %0 : !ty_22demo22, !cir.ptr<!ty_22demo22>
// CHECK: cir.return
// CHECK: }
// CHECK: cir.func no_proto @testTwo() extra(#fn_attr) {
// CHECK: %0 = cir.alloca !ty_22demo2E122, !cir.ptr<!ty_22demo2E122>, ["two"] {alignment = 4 : i64}
// CHECK: %1 = cir.const #cir.const_struct<{#cir.int<1> : !s32i, #cir.int<2> : !s32i}> : !ty_22demo2E122
// CHECK: cir.store %1, %0 : !ty_22demo2E122, !cir.ptr<!ty_22demo2E122>
// CHECK: cir.return
// CHECK: }
