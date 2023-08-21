// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Bar {
  int a;
  char b;
} bar;

struct Foo {
  int a;
  char b;
  struct Bar z;
};

void baz(void) {
  struct Bar b;
  struct Foo f;
}

// CHECK-DAG: !ty_22struct2EBar22 = !cir.struct<"struct.Bar", !s32i, !s8i>
// CHECK-DAG: !ty_22struct2EFoo22 = !cir.struct<"struct.Foo", !s32i, !s8i, !ty_22struct2EBar22>
//  CHECK-DAG: module {{.*}} {
     // CHECK:   cir.func @baz()
// CHECK-NEXT:     %0 = cir.alloca !ty_22struct2EBar22, cir.ptr <!ty_22struct2EBar22>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.alloca !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }

void shouldConstInitStructs(void) {
// CHECK: cir.func @shouldConstInitStructs
  struct Foo f = {1, 2, {3, 4}};
  // CHECK: %[[#V0:]] = cir.alloca !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
  // CHECK: %[[#V1:]] = cir.const(#cir.const_struct<{#cir.int<1> : !s32i, #cir.int<2> : !s8i, #cir.const_struct<{#cir.int<3> : !s32i, #cir.int<4> : !s8i}> : !ty_22struct2EBar22}> : !ty_22struct2EFoo22) : !ty_22struct2EFoo22
  // CHECK: cir.store %[[#V1]], %[[#V0]] : !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>
}

// Should zero-initialize uninitialized global structs.
struct S {
  int a,b;
} s;
// CHECK-DAG: cir.global external @s = #cir.zero : !ty_22struct2ES22

// Should initialize basic global structs.
struct S1 {
  int a;
  float f;
  int *p;
} s1 = {1, .1, 0};
// CHECK-DAG: cir.global external @s1 = #cir.const_struct<{#cir.int<1> : !s32i, 1.000000e-01 : f32, #cir.null : !cir.ptr<!s32i>}> : !ty_22struct2ES122

// Should initialize global nested structs.
struct S2 {
  struct S2A {
    int a;
  } s2a;
} s2 = {{1}};
// CHECK-DAG: cir.global external @s2 = #cir.const_struct<{#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_22struct2ES2A22}> : !ty_22struct2ES222

// Should initialize global arrays of structs.
struct S3 {
  int a;
} s3[3] = {{1}, {2}, {3}};
// CHECK-DAG: cir.global external @s3 = #cir.const_array<[#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_22struct2ES322, #cir.const_struct<{#cir.int<2> : !s32i}> : !ty_22struct2ES322, #cir.const_struct<{#cir.int<3> : !s32i}> : !ty_22struct2ES322]> : !cir.array<!ty_22struct2ES322 x 3>

void shouldCopyStructAsCallArg(struct S1 s) {
// CHECK-DAG: cir.func @shouldCopyStructAsCallArg
  shouldCopyStructAsCallArg(s);
  // CHECK-DAG: %[[#LV:]] = cir.load %{{.+}} : cir.ptr <!ty_22struct2ES122>, !ty_22struct2ES122
  // CHECK-DAG: cir.call @shouldCopyStructAsCallArg(%[[#LV]]) : (!ty_22struct2ES122) -> ()
}
