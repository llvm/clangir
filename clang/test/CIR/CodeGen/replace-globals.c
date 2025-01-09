// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

//#include <stdio.h>

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


// void check() {
//   printf("check: %d\n",****g5);
// }


int use() {
    int a = **g3;
    int b = ***g4; 
    int c = ****g5; 
    int d = **g8;    

    // printf("a: %d, b: %d, c: %d, d: %d\n", a,b,c,d);
    //check();

    return a + b + c + d;
}


int main() {
    use();
    return 0;
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


