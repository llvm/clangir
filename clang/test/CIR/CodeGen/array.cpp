// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir-enable -Wno-return-stack-address -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void a0() {
  int a[10];
}

// CHECK: cir.func @_Z2a0v() {
// CHECK-NEXT:   %0 = cir.alloca !cir.array<!s32i x 10>, cir.ptr <!cir.array<!s32i x 10>>, ["a"] {alignment = 16 : i64}

void a1() {
  int a[10];
  a[0] = 1;
}

// CHECK: cir.func @_Z2a1v() {
// CHECK-NEXT:  %0 = cir.alloca !cir.array<!s32i x 10>, cir.ptr <!cir.array<!s32i x 10>>, ["a"] {alignment = 16 : i64}
// CHECK-NEXT:  %1 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:  %2 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:  %3 = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<!s32i x 10>>), !cir.ptr<!s32i>
// CHECK-NEXT:  %4 = cir.ptr_stride(%3 : !cir.ptr<!s32i>, %2 : !s32i), !cir.ptr<!s32i>
// CHECK-NEXT:  cir.store %1, %4 : !s32i, cir.ptr <!s32i>

int *a2() {
  int a[4];
  return &a[0];
}

// CHECK: cir.func @_Z2a2v() -> !cir.ptr<!s32i> {
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.array<!s32i x 4>, cir.ptr <!cir.array<!s32i x 4>>, ["a"] {alignment = 16 : i64}
// CHECK-NEXT:   %2 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:   %3 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s32i x 4>>), !cir.ptr<!s32i>
// CHECK-NEXT:   %4 = cir.ptr_stride(%3 : !cir.ptr<!s32i>, %2 : !s32i), !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store %4, %0 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:   %5 = cir.load %0 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.return %5 : !cir.ptr<!s32i>

void local_stringlit() {
  const char *s = "whatnow";
}

// CHECK: cir.global "private" constant internal @".str" = #cir.const_array<"whatnow\00" : !cir.array<!s8i x 8>> : !cir.array<!s8i x 8> {alignment = 1 : i64}
// CHECK: cir.func @_Z15local_stringlitv() {
// CHECK-NEXT:  %0 = cir.alloca !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>, ["s", init] {alignment = 8 : i64}
// CHECK-NEXT:  %1 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 8>>
// CHECK-NEXT:  %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s8i x 8>>), !cir.ptr<!s8i>
// CHECK-NEXT:  cir.store %2, %0 : !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>
