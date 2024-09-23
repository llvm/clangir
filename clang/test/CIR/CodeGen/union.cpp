// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct { int x; } yolo;
typedef union { yolo y; struct { int lifecnt; }; } yolm;
typedef union { yolo y; struct { int *lifecnt; int genpad; }; } yolm2;
typedef union { yolo y; struct { bool life; int genpad; }; } yolm3;

// CHECK-DAG: !ty_U23A3ADummy = !cir.struct<struct "U2::Dummy" {!cir.int<s, 16>, !cir.float} #cir.record.decl.ast>
// CHECK-DAG: !ty_anon2E0_ = !cir.struct<struct "anon.0" {!cir.int<s, 32>} #cir.record.decl.ast>
// CHECK-DAG: !ty_anon2E2_ = !cir.struct<struct "anon.2" {!cir.bool, !cir.int<s, 32>} #cir.record.decl.ast>
// CHECK-DAG: !ty_yolo = !cir.struct<struct "yolo" {!cir.int<s, 32>} #cir.record.decl.ast>
// CHECK-DAG: !ty_anon2E1_ = !cir.struct<struct "anon.1" {!cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>} #cir.record.decl.ast>

// CHECK-DAG: !ty_yolm = !cir.struct<union "yolm" {!cir.struct<struct "yolo" {!cir.int<s, 32>} #cir.record.decl.ast>, !cir.struct<struct "anon.0" {!cir.int<s, 32>} #cir.record.decl.ast>}>
// CHECK-DAG: !ty_yolm3_ = !cir.struct<union "yolm3" {!cir.struct<struct "yolo" {!cir.int<s, 32>} #cir.record.decl.ast>, !cir.struct<struct "anon.2" {!cir.bool, !cir.int<s, 32>} #cir.record.decl.ast>}>
// CHECK-DAG: !ty_yolm2_ = !cir.struct<union "yolm2" {!cir.struct<struct "yolo" {!cir.int<s, 32>} #cir.record.decl.ast>, !cir.struct<struct "anon.1" {!cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>} #cir.record.decl.ast>}>

// Should generate a union type with all members preserved.
union U {
  bool b;
  short s;
  int i;
  float f;
  double d;
};
// CHECK-DAG: !ty_U = !cir.struct<union "U" {!cir.bool, !cir.int<s, 16>, !cir.int<s, 32>, !cir.float, !cir.double}>

// Should generate unions with complex members.
union U2 {
  bool b;
  struct Dummy {
    short s;
    float f;
  } s;
} u2;
// CHECK-DAG: !cir.struct<union "U2" {!cir.bool, !cir.struct<struct "U2::Dummy" {!cir.int<s, 16>, !cir.float} #cir.record.decl.ast>} #cir.record.decl.ast>

// Should genereate unions without padding.
union U3 {
  short b;
  U u;
} u3;
// CHECK-DAG: !ty_U3_ = !cir.struct<union "U3" {!cir.int<s, 16>, !cir.struct<union "U" {!cir.bool, !cir.int<s, 16>, !cir.int<s, 32>, !cir.float, !cir.double}>} #cir.record.decl.ast>

void m() {
  yolm q;
  yolm2 q2;
  yolm3 q3;
}

// CHECK:   cir.func @_Z1mv()
// CHECK:   cir.alloca !ty_yolm, !cir.ptr<!ty_yolm>, ["q"] {alignment = 4 : i64}
// CHECK:   cir.alloca !ty_yolm2_, !cir.ptr<!ty_yolm2_>, ["q2"] {alignment = 8 : i64}
// CHECK:   cir.alloca !ty_yolm3_, !cir.ptr<!ty_yolm3_>, ["q3"] {alignment = 4 : i64}

void shouldGenerateUnionAccess(union U u) {
  u.b = true;
  // CHECK: %[[#BASE:]] = cir.get_member %0[0] {name = "b"} : !cir.ptr<!ty_U> -> !cir.ptr<!cir.bool>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : !cir.bool, !cir.ptr<!cir.bool>
  u.b;
  // CHECK: cir.get_member %0[0] {name = "b"} : !cir.ptr<!ty_U> -> !cir.ptr<!cir.bool>
  u.i = 1;
  // CHECK: %[[#BASE:]] = cir.get_member %0[2] {name = "i"} : !cir.ptr<!ty_U> -> !cir.ptr<!s32i>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : !s32i, !cir.ptr<!s32i>
  u.i;
  // CHECK: %[[#BASE:]] = cir.get_member %0[2] {name = "i"} : !cir.ptr<!ty_U> -> !cir.ptr<!s32i>
  u.f = 0.1F;
  // CHECK: %[[#BASE:]] = cir.get_member %0[3] {name = "f"} : !cir.ptr<!ty_U> -> !cir.ptr<!cir.float>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : !cir.float, !cir.ptr<!cir.float>
  u.f;
  // CHECK: %[[#BASE:]] = cir.get_member %0[3] {name = "f"} : !cir.ptr<!ty_U> -> !cir.ptr<!cir.float>
  u.d = 0.1;
  // CHECK: %[[#BASE:]] = cir.get_member %0[4] {name = "d"} : !cir.ptr<!ty_U> -> !cir.ptr<!cir.double>
  // CHECK: cir.store %{{.+}}, %[[#BASE]] : !cir.double, !cir.ptr<!cir.double>
  u.d;
  // CHECK: %[[#BASE:]] = cir.get_member %0[4] {name = "d"} : !cir.ptr<!ty_U> -> !cir.ptr<!cir.double>
}

typedef union {
  short a;
  int b;
} A;

void noCrushOnDifferentSizes() {
  A a = {0};
  // CHECK:  %[[#TMP0:]] = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["a"] {alignment = 4 : i64}
  // CHECK:  %[[#TMP1:]] = cir.cast(bitcast, %[[#TMP0]] : !cir.ptr<!ty_A>), !cir.ptr<!ty_anon_struct>
  // CHECK:  %[[#TMP2:]] = cir.const #cir.zero : !ty_anon_struct
  // CHECK:  cir.store %[[#TMP2]], %[[#TMP1]] : !ty_anon_struct, !cir.ptr<!ty_anon_struct>
}