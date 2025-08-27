// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=OGCG

// Should generate basic pointer arithmetics.
void foo(int *iptr, char *cptr, unsigned ustride) {
  *(iptr + 2) = 1;
  // CIR: %[[#STRIDE:]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#STRIDE]] : !s32i, inbounds), !cir.ptr<!s32i>
  // LLVM: getelementptr inbounds
  // OGCG: getelementptr inbounds
  *(cptr + 3) = 1;
  // CIR: %[[#STRIDE:]] = cir.const #cir.int<3> : !s32i
  // CIR: cir.ptr_stride(%{{.+}} : !cir.ptr<!s8i>, %[[#STRIDE]] : !s32i, inbounds), !cir.ptr<!s8i>
  // LLVM: getelementptr inbounds
  // OGCG: getelementptr inbounds
  *(iptr - 2) = 1;
  // CIR: %[[#STRIDE:]] = cir.const #cir.int<2> : !s32i
  // CIR: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CIR: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#NEGSTRIDE]] : !s32i, inbounds), !cir.ptr<!s32i>
  // LLVM: getelementptr inbounds
  // OGCG: getelementptr inbounds
  *(cptr - 3) = 1;
  // CIR: %[[#STRIDE:]] = cir.const #cir.int<3> : !s32i
  // CIR: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#STRIDE]]) : !s32i, !s32i
  // CIR: cir.ptr_stride(%{{.+}} : !cir.ptr<!s8i>, %[[#NEGSTRIDE]] : !s32i, inbounds), !cir.ptr<!s8i>
  // LLVM: getelementptr inbounds
  // OGCG: getelementptr inbounds
  *(iptr + ustride) = 1;
  // CIR: %[[#STRIDE:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
  // CIR: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#STRIDE]] : !u32i, inbounds|nuw), !cir.ptr<!s32i>

  // LLVM: getelementptr inbounds nuw
  // OGCG: getelementptr inbounds nuw

  // Must convert unsigned stride to a signed one.
  *(iptr - ustride) = 1;
  // CIR: %[[#STRIDE:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!u32i>, !u32i
  // CIR: %[[#SIGNSTRIDE:]] = cir.cast(integral, %[[#STRIDE]] : !u32i), !s32i
  // CIR: %[[#NEGSTRIDE:]] = cir.unary(minus, %[[#SIGNSTRIDE]]) : !s32i, !s32i
  // CIR: cir.ptr_stride(%{{.+}} : !cir.ptr<!s32i>, %[[#NEGSTRIDE]] : !s32i, inbounds), !cir.ptr<!s32i>
  // LLVM: getelementptr inbounds
  // OGCG: getelementptr inbounds
}

void testPointerSubscriptAccess(int *ptr) {
// CIR: testPointerSubscriptAccess
  ptr[1] = 2;
  // CIR: %[[#V1:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[#V2:]] = cir.const #cir.int<1> : !s32i
  // CIR: cir.ptr_stride(%[[#V1]] : !cir.ptr<!s32i>, %[[#V2]] : !s32i), !cir.ptr<!s32i>
}

void testPointerMultiDimSubscriptAccess(int **ptr) {
// CIR: testPointerMultiDimSubscriptAccess
  ptr[1][2] = 3;
  // CIR: %[[#V1:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[#V2:]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[#V3:]] = cir.ptr_stride(%[[#V1]] : !cir.ptr<!cir.ptr<!s32i>>, %[[#V2]] : !s32i), !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[#V4:]] = cir.load{{.*}} %[[#V3]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: %[[#V5:]] = cir.const #cir.int<2> : !s32i
  // CIR: cir.ptr_stride(%[[#V4]] : !cir.ptr<!s32i>, %[[#V5]] : !s32i), !cir.ptr<!s32i>
}
