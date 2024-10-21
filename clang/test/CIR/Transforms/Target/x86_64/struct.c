// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir-flat -mmlir --mlir-print-ir-after=cir-call-conv-lowering %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct {
  int a, b;
} S;

// CHECK: cir.func @init(%arg0: !u64i
// CHECK: %[[#V0:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, [""] {alignment = 4 : i64}
// CHECK: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>
// CHECK: %[[#V2:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["__retval"] {alignment = 4 : i64}
// CHECK: %[[#V3:]] = cir.const #cir.int<1> : !s32i
// CHECK: %[[#V4:]] = cir.get_member %[[#V0]][0] {name = "a"} : !cir.ptr<!ty_S> -> !cir.ptr<!s32i>
// CHECK: cir.store %[[#V3]], %[[#V4]] : !s32i, !cir.ptr<!s32i>
// CHECK: %[[#V5:]] = cir.const #cir.int<2> : !s32i
// CHECK: %[[#V6:]] = cir.get_member %[[#V0]][1] {name = "b"} : !cir.ptr<!ty_S> -> !cir.ptr<!s32i>
// CHECK: cir.store %[[#V5]], %[[#V6]] : !s32i, !cir.ptr<!s32i>
// CHECK: cir.copy %[[#V0]] to %[[#V2]] : !cir.ptr<!ty_S>
// CHECK: %[[#V7:]] = cir.cast(bitcast, %[[#V2]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: %[[#V8:]] = cir.load %[[#V7]] : !cir.ptr<!u64i>, !u64i
// CHECK: cir.return %[[#V8]] : !u64i
S init(S s) {
  s.a = 1;
  s.b = 2;
  return s;
}

// CHECK: cir.func no_proto  @foo1
// CHECK: %[[#V0:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["s"]
// CHECK: %[[#V1:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["tmp"] {alignment = 4 : i64}
// CHECK: %[[#V2:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: %[[#V3:]] = cir.load %[[#V2]] : !cir.ptr<!u64i>, !u64i
// CHECK: %[[#V4:]] = cir.call @init(%[[#V3]]) : (!u64i) -> !u64i
// CHECK: %[[#V5:]] = cir.cast(bitcast, %[[#V1]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: cir.store %[[#V4]], %[[#V5]] : !u64i, !cir.ptr<!u64i>
// CHECK: cir.copy %[[#V1]] to %[[#V0]] : !cir.ptr<!ty_S>
// CHECK: cir.return
void foo1() {
  S s;
  s = init(s);
}

// CHECK: cir.func @foo2(%arg0: !u64i
// CHECK: %[[#V0:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, [""] {alignment = 4 : i64}
// CHECK: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: cir.store %arg0, %[[#V1]] : !u64i, !cir.ptr<!u64i>
// CHECK: %[[#V2:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["__retval"] {alignment = 4 : i64}
// CHECK: %[[#V3:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["s2"]
// CHECK: %[[#V4:]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["tmp"] {alignment = 4 : i64}
// CHECK: %[[#V5:]] = cir.const #cir.const_struct<{#cir.int<1> : !s32i, #cir.int<2> : !s32i}> : !ty_S
// CHECK: cir.store %[[#V5]], %[[#V3]] : !ty_S, !cir.ptr<!ty_S>
// CHECK: %[[#V6:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: %[[#V7:]] = cir.load %[[#V6]] : !cir.ptr<!u64i>, !u64i
// CHECK: %[[#V8:]] = cir.call @foo2(%[[#V7]]) : (!u64i) -> !u64i
// CHECK: %[[#V9:]] = cir.cast(bitcast, %[[#V4]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: cir.store %[[#V8]], %[[#V9]] : !u64i, !cir.ptr<!u64i>
// CHECK: cir.copy %[[#V4]] to %[[#V0]] : !cir.ptr<!ty_S>
// CHECK: cir.copy %[[#V0]] to %[[#V2]] : !cir.ptr<!ty_S>
// CHECK: %[[#V10:]] = cir.cast(bitcast, %[[#V2]] : !cir.ptr<!ty_S>), !cir.ptr<!u64i>
// CHECK: %[[#V11:]] = cir.load %[[#V10]] : !cir.ptr<!u64i>, !u64i
// CHECK: cir.return %[[#V11]] : !u64i
S foo2(S s1) {
  S s2 = {1, 2};
  s1 = foo2(s1);
  return s1;
}

typedef struct {
  char a;
  char b;
} S2;

// CHECK: cir.func @init2(%arg0: !u16i
// CHECK: %[[#V0:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, [""] {alignment = 4 : i64}
// CHECK: %[[#V1:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S2_>), !cir.ptr<!u16i>
// CHECK: cir.store %arg0, %[[#V1]] : !u16i, !cir.ptr<!u16i>
// CHECK: %[[#V2:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, ["__retval"] {alignment = 1 : i64}
// CHECK: %[[#V3:]] = cir.const #cir.int<1> : !s32i
// CHECK: %[[#V4:]] = cir.cast(integral, %[[#V3]] : !s32i), !s8i
// CHECK: %[[#V5:]] = cir.get_member %[[#V0]][0] {name = "a"} : !cir.ptr<!ty_S2_> -> !cir.ptr<!s8i>
// CHECK: cir.store %[[#V4]], %[[#V5]] : !s8i, !cir.ptr<!s8i>
// CHECK: %[[#V6:]] = cir.const #cir.int<2> : !s32i
// CHECK: %[[#V7:]] = cir.cast(integral, %[[#V6]] : !s32i), !s8i
// CHECK: %[[#V8:]] = cir.get_member %[[#V0]][1] {name = "b"} : !cir.ptr<!ty_S2_> -> !cir.ptr<!s8i>
// CHECK: cir.store %[[#V7]], %[[#V8]] : !s8i, !cir.ptr<!s8i>
// CHECK: cir.copy %[[#V0]] to %[[#V2]] : !cir.ptr<!ty_S2_>
// CHECK: %[[#V9:]] = cir.cast(bitcast, %[[#V2]] : !cir.ptr<!ty_S2_>), !cir.ptr<!u16i>
// CHECK: %[[#V10:]] = cir.load %[[#V9]] : !cir.ptr<!u16i>, !u16i
// CHECK: cir.return %[[#V10]] : !u16i
S2 init2(S2 s) {
  s.a = 1;
  s.b = 2;
  return s;
}

// CHECK: cir.func no_proto  @foo3()
// CHECK: %[[#V0:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, ["s"]
// CHECK: %[[#V1:]] = cir.alloca !ty_S2_, !cir.ptr<!ty_S2_>, ["tmp"] {alignment = 1 : i64}
// CHECK: %[[#V2:]] = cir.cast(bitcast, %[[#V0]] : !cir.ptr<!ty_S2_>), !cir.ptr<!u16i>
// CHECK: %[[#V3:]] = cir.load %[[#V2]] : !cir.ptr<!u16i>, !u16i
// CHECK: %[[#V4:]] = cir.call @init2(%[[#V3]]) : (!u16i) -> !u16i
// CHECK: %[[#V5:]] = cir.cast(bitcast, %[[#V1]] : !cir.ptr<!ty_S2_>), !cir.ptr<!u16i>
// CHECK: cir.store %[[#V4]], %[[#V5]] : !u16i, !cir.ptr<!u16i>
// CHECK: cir.copy %[[#V1]] to %[[#V0]] : !cir.ptr<!ty_S2_>
// CHECK: cir.return
void foo3() {
  S2 s;
  s = init2(s);
}