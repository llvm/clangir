// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct { int x; } yolo;
typedef union { yolo y; struct { int lifecnt; }; } yolm;
typedef union { yolo y; struct { int *lifecnt; int genpad; }; } yolm2;
typedef union { yolo y; struct { bool life; int genpad; }; } yolm3;

// CHECK-DAG: !ty_22struct2EU23A3ADummy22 = !cir.struct<struct "struct.U2::Dummy" {!s16i, f32} #cir.recdecl.ast>
// CHECK-DAG: !ty_22struct2Eanon221 = !cir.struct<struct "struct.anon" {!cir.bool, !s32i} #cir.recdecl.ast>
// CHECK-DAG: !ty_22struct2Eyolo22 = !cir.struct<struct "struct.yolo" {!s32i} #cir.recdecl.ast>
// CHECK-DAG: !ty_22struct2Eanon222 = !cir.struct<struct "struct.anon" {!cir.ptr<!s32i>, !s32i} #cir.recdecl.ast>

// CHECK-DAG: !ty_22union2Eyolm22 = !cir.struct<union "union.yolm" {!ty_22struct2Eyolo22, !ty_22struct2Eanon22}>
// CHECK-DAG: !ty_22union2Eyolm322 = !cir.struct<union "union.yolm3" {!ty_22struct2Eyolo22, !ty_22struct2Eanon221}>
// CHECK-DAG: !ty_22union2Eyolm222 = !cir.struct<union "union.yolm2" {!ty_22struct2Eyolo22, !ty_22struct2Eanon222}>

// Should generate a union type with all members preserved.
union U {
  bool b;
  short s;
  int i;
  float f;
  double d;
};
// CHECK-DAG: !ty_22union2EU22 = !cir.struct<union "union.U" {!cir.bool, !s16i, !s32i, f32, f64}>

// Should generate unions with complex members.
union U2 {
  bool b;
  struct Dummy {
    short s;
    float f;
  } s;
} u2;
// CHECK-DAG: !cir.struct<union "union.U2" {!cir.bool, !ty_22struct2EU23A3ADummy22} #cir.recdecl.ast>

// Should genereate unions without padding.
union U3 {
  short b;
  U u;
} u3;
// CHECK-DAG: !ty_22union2EU322 = !cir.struct<union "union.U3" {!s16i, !ty_22union2EU22} #cir.recdecl.ast>

void m() {
  yolm q;
  yolm2 q2;
  yolm3 q3;
}

// CHECK:   cir.func @_Z1mv()
// CHECK:   cir.alloca !ty_22union2Eyolm22, cir.ptr <!ty_22union2Eyolm22>, ["q"] {alignment = 4 : i64}
// CHECK:   cir.alloca !ty_22union2Eyolm222, cir.ptr <!ty_22union2Eyolm222>, ["q2"] {alignment = 8 : i64}
// CHECK:   cir.alloca !ty_22union2Eyolm322, cir.ptr <!ty_22union2Eyolm322>, ["q3"] {alignment = 4 : i64}

void shouldGenerateUnionAccess(union U u) {
  u.b = true;
  // CHECK: %[[#BASE:]] = cir.get_member %0[0] {name = "b"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<!cir.bool>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : !cir.bool, cir.ptr <!cir.bool>
  u.b;
  // CHECK: cir.get_member %0[0] {name = "b"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<!cir.bool>
  u.i = 1;
  // CHECK: %[[#BASE:]] = cir.get_member %0[2] {name = "i"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<!s32i>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : !s32i, cir.ptr <!s32i>
  u.i;
  // CHECK: %[[#BASE:]] = cir.get_member %0[2] {name = "i"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<!s32i>
  u.f = 0.1F;
  // CHECK: %[[#BASE:]] = cir.get_member %0[3] {name = "f"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<f32>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : f32, cir.ptr <f32>
  u.f;
  // CHECK: %[[#BASE:]] = cir.get_member %0[3] {name = "f"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<f32>
  u.d = 0.1;
  // CHECK: %[[#BASE:]] = cir.get_member %0[4] {name = "d"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<f64>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : f64, cir.ptr <f64>
  u.d;
  // CHECK: %[[#BASE:]] = cir.get_member %0[4] {name = "d"} : !cir.ptr<!ty_22union2EU22> -> !cir.ptr<f64>
}
