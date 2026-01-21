// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

#include <stdarg.h>

int f1(int n, ...) {
  va_list valist;
  va_start(valist, n);
  int res = va_arg(valist, int);
  va_end(valist);
  return res;
}

// CIR: !rec___va_list = !cir.record<struct "__va_list" {!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i}
// CIR:  cir.func {{.*}} @f1(%arg0: !s32i, ...) -> !s32i
// CIR:  [[RETP:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:  [[RESP:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["res", init]
// CIR:  cir.va_start [[VARLIST:%.*]] {{.*}} : !cir.ptr<!rec___va_list>, !s32i
// CIR:  [[TMP0:%.*]] = cir.va_arg [[VARLIST]] : (!cir.ptr<!rec___va_list>) -> !s32i
// CIR:  cir.store{{.*}} [[TMP0]], [[RESP]] : !s32i, !cir.ptr<!s32i>
// CIR:  cir.va_end [[VARLIST]] : !cir.ptr<!rec___va_list>
// CIR:  [[RES:%.*]] = cir.load{{.*}} [[RESP]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store{{.*}} [[RES]], [[RETP]] : !s32i, !cir.ptr<!s32i>
// CIR:  [[RETV:%.*]] = cir.load{{.*}} [[RETP]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return [[RETV]] : !s32i

// LLVM: %struct.__va_list = type { ptr, ptr, ptr, i32, i32 }
// LLVM: define dso_local i32 @f1(i32 %0, ...)
// LLVM: call void @llvm.va_start.p0(ptr [[VARLIST:%.*]])
// LLVM: [[TMP:%.*]] = va_arg ptr [[VARLIST]], i32
// LLVM: call void @llvm.va_end.p0(ptr [[VARLIST]])
// LLVM: ret i32
