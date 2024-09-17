// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct {
  int a;
  int b[2];
} A;

int bar() {
  return 42;
}

void foo() {
  A a = {bar(), {}};
}
// CHECK: %0 = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["a", init]
// CHECK: %1 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init]
// CHECK: %2 = cir.get_member %0[0] {name = "a"} : !cir.ptr<!ty_A> -> !cir.ptr<!s32i>
// CHECK: %3 = cir.call @_Z3barv() : () -> !s32i
// CHECK: cir.store %3, %2 : !s32i, !cir.ptr<!s32i>
// CHECK: %4 = cir.get_member %0[1] {name = "b"} : !cir.ptr<!ty_A> -> !cir.ptr<!cir.array<!s32i x 2>>
// CHECK: %5 = cir.cast(array_to_ptrdecay, %4 : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
// CHECK: cir.store %5, %1 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %6 = cir.const #cir.int<2> : !s64i
// CHECK: %7 = cir.ptr_stride(%5 : !cir.ptr<!s32i>, %6 : !s64i), !cir.ptr<!s32i>
// CHECK: cir.do {
// CHECK:     %8 = cir.load %1 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %9 = cir.const #cir.int<0> : !s32i
// CHECK:     cir.store %9, %8 : !s32i, !cir.ptr<!s32i>
// CHECK:     %10 = cir.const #cir.int<1> : !s64i
// CHECK:     %11 = cir.ptr_stride(%8 : !cir.ptr<!s32i>, %10 : !s64i), !cir.ptr<!s32i>
// CHECK:     cir.store %11, %1 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:     cir.yield
// CHECK: } while {
// CHECK:     %8 = cir.load %1 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %9 = cir.cmp(ne, %8, %7) : !cir.ptr<!s32i>, !cir.bool
// CHECK:     cir.condition(%9)
// CHECK: }