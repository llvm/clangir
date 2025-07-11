// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

typedef struct {
  int x;
} A;

int cstyle_cast_lvalue(A a) {
  return ((A)(a)).x;
}

// CHECK:  cir.func dso_local @cstyle_cast_lvalue(%arg0: !rec_A loc({{.*}}))
// CHECK:    [[ALLOC_A:%.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a", init] {alignment = 4 : i64}
// CHECK:    [[ALLOC_RET:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:    cir.store{{.*}} %arg0, [[ALLOC_A]] : !rec_A, !cir.ptr<!rec_A>
// CHECK:    [[X_ADDR:%.*]] = cir.get_member [[ALLOC_A]][0] {name = "x"} : !cir.ptr<!rec_A> -> !cir.ptr<!s32i>
// CHECK:    [[X:%.*]] = cir.load{{.*}} [[X_ADDR]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.store{{.*}} [[X]], [[ALLOC_RET]] : !s32i, !cir.ptr<!s32i>
// CHECK:    [[RET:%.*]] = cir.load{{.*}} [[ALLOC_RET]] : !cir.ptr<!s32i>, !s32i
// CHECK:    cir.return [[RET]] : !s32i
