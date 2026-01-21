// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

void bar() {
  const int arr[1] = {1};
}

// CHECK-LABEL: @bar()
// CHECK: %[[ADDR:.*]] = cir.alloca !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>, ["arr", init, const]
// CHECK: %[[VAL:.*]] = cir.get_global @__const.bar.arr : !cir.ptr<!cir.array<!s32i x 1>>
// CHECK: cir.copy %[[VAL]] to %[[ADDR]] : !cir.ptr<!cir.array<!s32i x 1>>

void foo() {
  int a[10] = {1};
}

// CHECK-LABEL: @foo()
// CHECK: %[[ADDR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["a"]
// CHECK: %[[CAST:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!rec_anon_struct>
// CHECK: %[[VAL:.*]] = cir.get_global @__const.foo.a : !cir.ptr<!rec_anon_struct>
// CHECK: cir.copy %[[VAL]] to %[[CAST]] : !cir.ptr<!rec_anon_struct>
