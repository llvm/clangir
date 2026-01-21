// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

#include <stdarg.h>

double f1(int n, ...) {
  va_list valist;
  va_start(valist, n);
  double res = va_arg(valist, double);
  va_end(valist);
  return res;
}

// CIR: !rec___va_list = !cir.record<struct "__va_list" {!cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i}
// CIR:  cir.func {{.*}} @f1(%arg0: !s32i, ...) -> !cir.double
// CIR:  [[RETP:%.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["__retval"]
// CIR:  [[RESP:%.*]] = cir.alloca !cir.double, !cir.ptr<!cir.double>, ["res", init]
// CIR:  cir.va_start [[VARLIST:%.*]] {{.*}} : !cir.ptr<!rec___va_list>, !s32i
// CIR:  [[TMP0:%.*]] = cir.va_arg [[VARLIST]] : (!cir.ptr<!rec___va_list>) -> !cir.double
// CIR:  cir.store{{.*}} [[TMP0]], [[RESP]] : !cir.double, !cir.ptr<!cir.double>
// CIR:  cir.va_end [[VARLIST]] : !cir.ptr<!rec___va_list>
// CIR:  [[RES:%.*]] = cir.load{{.*}} [[RESP]] : !cir.ptr<!cir.double>, !cir.double
// CIR:   cir.store{{.*}} [[RES]], [[RETP]] : !cir.double, !cir.ptr<!cir.double>
// CIR:  [[RETV:%.*]] = cir.load{{.*}} [[RETP]] : !cir.ptr<!cir.double>, !cir.double
// CIR:   cir.return [[RETV]] : !cir.double

// LLVM: %struct.__va_list = type { ptr, ptr, ptr, i32, i32 }
// LLVM: define dso_local double @f1(i32 %0, ...)
// LLVM: call void @llvm.va_start.p0(ptr [[VARLIST:%.*]])
// LLVM: [[TMP:%.*]] = va_arg ptr [[VARLIST]], double
// LLVM: call void @llvm.va_end.p0(ptr [[VARLIST]])
// LLVM: ret double
