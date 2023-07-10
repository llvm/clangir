// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void foo(int *iptr, char *cptr) {
  iptr + 2;
  // CHECK: %[[#STRIDE2:]] = cir.const(#cir.int<2> : !s32i) : !s32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#STRIDE2]] : !s32i), !cir.ptr<!s32i>
  cptr + 3;
  // CHECK: %[[#STRIDE3:]] = cir.const(#cir.int<3> : !s32i) : !s32i
  // CHECK: cir.ptr_stride(%{{.+}} : !cir.ptr<!s8i>, %[[#STRIDE3]] : !s32i), !cir.ptr<!s8i>
}
