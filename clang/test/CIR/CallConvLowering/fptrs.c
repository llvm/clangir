// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir-flat -fclangir-call-conv-lowering %s -o - | FileCheck %s

typedef struct {
  int a;
} S;

typedef int (*myfptr)(S);

int foo(S s) { return 42 + s.a; }

// CHECK: cir.func {{.*@bar}}
// CHECK:   %0 = cir.alloca !cir.ptr<!cir.func<!s32i (!s32i)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!s32i)>>>, ["a"]
// CHECK:   %1 = cir.get_global @foo : !cir.ptr<!cir.func<!s32i (!s32i)>>
// CHECK:   cir.store %1, %0 : !cir.ptr<!cir.func<!s32i (!s32i)>>, !cir.ptr<!cir.ptr<!cir.func<!s32i (!s32i)>>>
void bar() {
  myfptr a = foo;
}
