// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

void bar() {
  const int arr[1] = {1};
}

// CHECK: cir.global "private" constant internal dso_local @bar.arr = #cir.const_array<[#cir.int<1> : !s32i]> : !cir.array<!s32i x 1> {alignment = 4 : i64}
// CHECK: cir.func no_proto dso_local @bar()
// CHECK:   {{.*}} = cir.get_global @bar.arr : !cir.ptr<!cir.array<!s32i x 1>>

void foo() {
  int a[10] = {1};
}

// CHECK-LABEL: @foo()
// CHECK: %[[ADDR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["a"]
// CHECK: %[[SRC:.*]] = cir.get_global @__const.foo.a : !cir.ptr<!cir.array<!s32i x 10>>
// CHECK: cir.copy %[[SRC]] to %[[ADDR]] : !cir.ptr<!cir.array<!s32i x 10>>
