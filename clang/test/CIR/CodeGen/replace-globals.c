// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OG

typedef struct {
   int f0 : 24;
   int f1;
   int f2;
} S;

static S g1 = {2799, 9, 123};
static int *g2[4] = {&g1.f1, &g1.f1, &g1.f1, &g1.f1};
static int **g3 = &g2[1];
static int ***g4 = &g3;
static int ****g5 = &g4;

static S g6[2] = {{2799, 9, 123}, {2799, 9, 123}};
static int *g7[2] = {&g6[0].f2, &g6[1].f2};
static int **g8 = &g7[1];

void use() {
    int a = **g3;
    int b = ***g4; 
    int c = ****g5; 
    int d = **g8;    
}

// CHECK-DAG: !ty_anon_struct = !cir.struct<struct  {!u8i, !u8i, !u8i, !u8i, !s32i, !s32i}>
// CHECK-DAG: g1 = #cir.const_struct<{#cir.int<239> : !u8i, #cir.int<10> : !u8i, #cir.int<0> : !u8i, #cir.zero : !u8i, #cir.int<9> : !s32i, #cir.int<123> : !s32i}> : !ty_anon_struct
// CHECK-DAG: g2 = #cir.const_array<[#cir.global_view<@g1, [4]> : !cir.ptr<!ty_anon_struct>, #cir.global_view<@g1, [4]> : !cir.ptr<!ty_anon_struct>, #cir.global_view<@g1, [4]> : !cir.ptr<!ty_anon_struct>, #cir.global_view<@g1, [4]> : !cir.ptr<!ty_anon_struct>]> : !cir.array<!cir.ptr<!s32i> x 4>
// CHECK-DAG: g3 = #cir.global_view<@g2, [1 : i32]> : !cir.ptr<!cir.ptr<!s32i>>
// CHECK-DAG: g4 = #cir.global_view<@g3> : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CHECK-DAG: g5 = #cir.global_view<@g4> : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>>
// CHECK-DAG: g6 = #cir.const_array<[#cir.const_struct<{#cir.int<239> : !u8i, #cir.int<10> : !u8i, #cir.int<0> : !u8i, #cir.zero : !u8i, #cir.int<9> : !s32i, #cir.int<123> : !s32i}> : !ty_anon_struct, #cir.const_struct<{#cir.int<239> : !u8i, #cir.int<10> : !u8i, #cir.int<0> : !u8i, #cir.zero : !u8i, #cir.int<9> : !s32i, #cir.int<123> : !s32i}> : !ty_anon_struct]> : !cir.array<!ty_anon_struct x 2> 
// CHECK-DAG: g7 = #cir.const_array<[#cir.global_view<@g6, [0, 5]> : !cir.ptr<!s32i>, #cir.global_view<@g6, [1, 5]> : !cir.ptr<!s32i>]> : !cir.array<!cir.ptr<!s32i> x 2> 
// CHECK-DAG: g8 = #cir.global_view<@g7, [1 : i32]> : !cir.ptr<!cir.ptr<!s32i>> 

// LLVM-DAG: @g1 = internal global { i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }, align 4
// LLVM-DAG: @g2 = internal global [4 x ptr] [ptr getelementptr inbounds ({ i8, i8, i8, i8, i32, i32 }, ptr @g1, i32 0, i32 4), ptr getelementptr inbounds ({ i8, i8, i8, i8, i32, i32 }, ptr @g1, i32 0, i32 4), ptr getelementptr inbounds ({ i8, i8, i8, i8, i32, i32 }, ptr @g1, i32 0, i32 4), ptr getelementptr inbounds ({ i8, i8, i8, i8, i32, i32 }, ptr @g1, i32 0, i32 4)], align 16
// LLVM-DAG: @g3 = internal global ptr getelementptr inbounds ([4 x ptr], ptr @g2, i32 0, i32 1), align 8
// LLVM-DAG: @g4 = internal global ptr @g3, align 8
// LLVM-DAG: @g5 = internal global ptr @g4, align 8
// LLVM-DAG: @g6 = internal global [2 x { i8, i8, i8, i8, i32, i32 }] [{ i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }, { i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }], align 16
// LLVM-DAG: @g7 = internal global [2 x ptr] [ptr getelementptr inbounds ([2 x { i8, i8, i8, i8, i32, i32 }], ptr @g6, i32 0, i32 0, i32 5), ptr getelementptr inbounds ([2 x { i8, i8, i8, i8, i32, i32 }], ptr @g6, i32 0, i32 1, i32 5)], align 16
// LLVM-DAG: @g8 = internal global ptr getelementptr inbounds ([2 x ptr], ptr @g7, i32 0, i32 1), align 8

// OG-DAG: @g1 = internal global { i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }, align 4
// OG-DAG: @g2 = internal global [4 x ptr] [ptr getelementptr (i8, ptr @g1, i64 4), ptr getelementptr (i8, ptr @g1, i64 4), ptr getelementptr (i8, ptr @g1, i64 4), ptr getelementptr (i8, ptr @g1, i64 4)], align 16
// OG-DAG: @g3 = internal global ptr getelementptr (i8, ptr @g2, i64 8), align 8
// OG-DAG: @g4 = internal global ptr @g3, align 8
// OG-DAG: @g5 = internal global ptr @g4, align 8
// OG-DAG: @g6 = internal global [2 x { i8, i8, i8, i8, i32, i32 }] [{ i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }, { i8, i8, i8, i8, i32, i32 } { i8 -17, i8 10, i8 0, i8 0, i32 9, i32 123 }], align 16
// OG-DAG: @g7 = internal global [2 x ptr] [ptr getelementptr (i8, ptr @g6, i64 8), ptr getelementptr (i8, ptr @g6, i64 20)], align 16
// OG-DAG: @g8 = internal global ptr getelementptr (i8, ptr @g7, i64 8), align 8






