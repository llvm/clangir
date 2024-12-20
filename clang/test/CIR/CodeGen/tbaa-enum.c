// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes -relaxed-aliasing
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O0 -disable-llvm-passes
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s

// NO-TBAA-NOT: !tbaa

// CIR: #tbaa[[tbaa_NYI:.*]] = #cir.tbaa
// CIR: #tbaa[[INT:.*]] = #cir.tbaa_scalar<type = !u32i>
// CIR: #tbaa[[INT_PTR:.*]] = #cir.tbaa_scalar<type = !cir.ptr<!u32i>>

typedef unsigned int uint32_t;
typedef enum {
  RED_AUTO_32,
  GREEN_AUTO_32,
  BLUE_AUTO_32
} EnumAuto32;

uint32_t g0(EnumAuto32 *E, uint32_t *val) {
  // CIR-LABEL: cir.func @g0
  // CIR: %[[C5:.*]] = cir.const #cir.int<5> : !s32i
  // CIR: %[[U_C5:.*]] = cir.cast(integral, %[[C5]] : !s32i), !u32i
  // CIR: %[[VAL_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i> tbaa(#tbaa[[INT_PTR]])
  // CIR: cir.store %[[U_C5]], %[[VAL_PTR]] : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[INT]])
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[U_C0:.*]] = cir.cast(integral, %[[C0]] : !s32i), !u32i
  // CIR: %[[E_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i> tbaa(#tbaa[[INT_PTR]])
  // CIR: cir.store %[[U_C0]], %[[E_PTR]] : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[tbaa_NYI]])
  // CIR: %[[RET_PTR:.*]] = cir.load deref %{{.*}} : !cir.ptr<!cir.ptr<!u32i>>, !cir.ptr<!u32i> tbaa(#tbaa[[INT_PTR]])
  // CIR: %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!u32i>, !u32i tbaa(#tbaa[[INT]])
  // CIR: cir.store %[[RET]], %{{.*}} : !u32i, !cir.ptr<!u32i>

  // LLVM-LABEL: define{{.*}} i32 @g0(
  // LLVM: store i32 5, ptr %{{.*}}, align 4, !tbaa [[TAG_i32:!.*]]
  // LLVM: store i32 0, ptr %{{.*}}, align 4
  // LLVM: load i32, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  *val = 5;
  *E = RED_AUTO_32;
  return *val;
}

// LLVM: [[TYPE_char:!.*]] = !{!"omnipotent char", [[TAG_c_tbaa:!.*]],
// LLVM: [[TAG_c_tbaa]] = !{!"Simple C/C++ TBAA"}
// LLVM: [[TAG_i32]] = !{[[TYPE_i32:!.*]], [[TYPE_i32]], i64 0}
// LLVM: [[TYPE_i32]] = !{!"int", [[TYPE_char]], i64 0}
