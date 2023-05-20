// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct { int x; } yolo;
typedef union { yolo y; struct { int lifecnt; }; } yolm;
typedef union { yolo y; struct { int *lifecnt; int genpad; }; } yolm2;
typedef union { yolo y; struct { bool life; int genpad; }; } yolm3;

void m() {
  yolm q;
  yolm2 q2;
  yolm3 q3;
}

// CHECK: !ty_22struct2Eanon22 = !cir.struct<"struct.anon", !cir.bool, !s32i, #cir.recdecl.ast>
// CHECK: !ty_22struct2Eyolo22 = !cir.struct<"struct.yolo", !s32i, #cir.recdecl.ast>
// CHECK: !ty_22struct2Eanon221 = !cir.struct<"struct.anon", !cir.ptr<!s32i>, !s32i, #cir.recdecl.ast>

// CHECK: !ty_22union2Eyolm22 = !cir.struct<"union.yolm", !ty_22struct2Eyolo22>
// CHECK: !ty_22union2Eyolm222 = !cir.struct<"union.yolm2", !ty_22struct2Eanon221>

// CHECK:   cir.func @_Z1mv() {
// CHECK:   cir.alloca !ty_22union2Eyolm22, cir.ptr <!ty_22union2Eyolm22>, ["q"] {alignment = 4 : i64}
// CHECK:   cir.alloca !ty_22union2Eyolm222, cir.ptr <!ty_22union2Eyolm222>, ["q2"] {alignment = 8 : i64}
// CHECK:   cir.alloca !ty_22union2Eyolm322, cir.ptr <!ty_22union2Eyolm322>, ["q3"] {alignment = 4 : i64} loc(#loc12)