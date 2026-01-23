// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

void bar() {
  const int arr[1] = {1};
}

// CHECK-LABEL: @bar()
// CHECK: %[[ADDR:.*]] = cir.alloca !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>, ["arr", init, const]
// CHECK: %[[VAL:.*]] = cir.const #cir.const_array<[#cir.int<1> : !s32i]> : !cir.array<!s32i x 1>
// CHECK: cir.store {{.*}} %[[VAL]], %[[ADDR]] : !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>

void foo() {
  int a[10] = {1};
}

// CHECK-LABEL: @foo()
// CHECK: %[[ADDR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["a"]
// CHECK: %[[CAST:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!rec_anon_struct>
// CHECK: %[[VAL:.*]] = cir.const #cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !cir.array<!s32i x 9>}> : !rec_anon_struct
// CHECK: cir.store {{.*}} %[[VAL]], %[[CAST]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
