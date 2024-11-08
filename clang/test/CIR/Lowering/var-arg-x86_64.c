// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

#include <stdarg.h>

double f1(int n, ...) {
  va_list valist;
  va_start(valist, n);
  double res = va_arg(valist, double);
  va_end(valist);
  return res;
}

// CHECK: [[VA_LIST_TYPE:%.+]] = type { i32, i32, ptr, ptr }

// CHECK: define {{.*}}@f1
// CHECK: [[VA_LIST_ALLOCA:%.+]] = alloca {{.*}}[[VA_LIST_TYPE]]
// CHECK: [[VA_LIST:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: call {{.*}}@llvm.va_start.p0(ptr [[VA_LIST]])
// CHECK: [[VA_LIST2:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: [[FP_OFFSET_P:%.+]] = getelementptr {{.*}} [[VA_LIST2]], i32 0, i32 1
// CHECK: [[FP_OFFSET:%.+]] = load {{.*}}, ptr [[FP_OFFSET_P]]
// CHECK: [[COMPARED:%.+]] = icmp ule i32 {{.*}}, 160
// CHECK: br i1 [[COMPARED]], label %[[THEN_BB:.+]], label %[[ELSE_BB:.+]],
//
// CHECK: [[THEN_BB]]:
// CHECK:   [[UPDATED_FP_OFFSET:%.+]] = add i32 [[FP_OFFSET]], 8
// CHECK:   store i32 [[UPDATED_FP_OFFSET]], ptr [[FP_OFFSET_P]]
// CHECK:   br label %[[CONT_BB:.+]],
//
// CHECK: [[ELSE_BB]]:
// CHECK:   [[OVERFLOW_ARG_AREA_ADDR:%.+]] = getelementptr {{.*}} [[VA_LIST2]], i32 0, i32 2
// CHECK:   [[OVERFLOW_ARG_AREA:%.+]] = load ptr, ptr [[OVERFLOW_ARG_AREA_ADDR]]
// CHECK:   [[OVERFLOW_ARG_AREA_OFFSET:%.+]] = getelementptr {{.*}} [[OVERFLOW_ARG_AREA]], i64 8
// CHECK:   store ptr [[OVERFLOW_ARG_AREA_OFFSET]], ptr [[OVERFLOW_ARG_AREA_ADDR]]
// CHECK:   br label %[[CONT_BB]]
//
// CHECK: [[CONT_BB]]:
// CHECK: [[VA_LIST3:%.+]] = getelementptr {{.*}} [[VA_LIST_ALLOCA]], i32 0
// CHECK: call {{.*}}@llvm.va_end.p0(ptr [[VA_LIST3]])
