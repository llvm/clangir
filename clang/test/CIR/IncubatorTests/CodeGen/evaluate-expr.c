// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

static const int g = 1;
void foo() {
  if ((g != 1) && (g != 1))
    return;
  if ((g == 1) || (g == 1))
    return;
}
// Upstream generates full ternary operations instead of constant folding.
// CHECK:  cir.func {{.*}} @foo()
// CHECK:    cir.scope {
// CHECK:      %[[G1:.*]] = cir.get_global @g
// CHECK:      %[[VAL1:.*]] = cir.load{{.*}} %[[G1]]
// CHECK:      %[[ONE1:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:      %[[CMP1:.*]] = cir.cmp(ne, %[[VAL1]], %[[ONE1]])
// CHECK:      %[[TERN1:.*]] = cir.ternary(%[[CMP1]], true
// CHECK:      cir.if %[[TERN1]]
// CHECK:    }
// CHECK:    cir.scope {
// CHECK:      %[[G2:.*]] = cir.get_global @g
// CHECK:      %[[VAL2:.*]] = cir.load{{.*}} %[[G2]]
// CHECK:      %[[ONE2:.*]] = cir.const #cir.int<1> : !s32i
// CHECK:      %[[CMP2:.*]] = cir.cmp(eq, %[[VAL2]], %[[ONE2]])
// CHECK:      %[[TERN2:.*]] = cir.ternary(%[[CMP2]], true
// CHECK:      cir.if %[[TERN2]]
// CHECK:    }
// CHECK:    cir.return

typedef struct { int x; } S;
static const S s = {0};
void bar() {
  int a =  s.x;
}
// CHECK:  cir.func {{.*}} @bar()
// CHECK:    [[ALLOC:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// CHECK:    {{%.*}} = cir.get_global @s : !cir.ptr<!rec_S>
// CHECK:    [[CONST:%.*]] = cir.const #cir.int<0> : !s32i
// CHECK:    cir.store{{.*}} [[CONST]], [[ALLOC]] : !s32i, !cir.ptr<!s32i>
// CHECK:    cir.return
