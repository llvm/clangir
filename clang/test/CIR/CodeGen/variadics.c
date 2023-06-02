// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include <stdarg.h>
// CHECK: [[VALISTTYPE:!.+va_list_.+]] = !cir.struct<"struct.__va_list_tag"

int average(int count, ...) {
// CHECK: cir.func @{{.*}}average{{.*}}(%arg0: !s32i loc({{.+}}), ...) -> !s32i
    va_list args, args_copy;
    va_start(args, count);
    // CHECK: cir.vastart %{{[0-9]+}} : !cir.ptr<[[VALISTTYPE]]>

    va_copy(args_copy, args);
    // CHECK: cir.vacopy %{{[0-9]+}} to %{{[0-9]+}} : !cir.ptr<[[VALISTTYPE]]>, !cir.ptr<[[VALISTTYPE]]>

    int sum = 0;
    for(int i = 0; i < count; i++) {
        sum += va_arg(args, int);
        // CHECK: %{{[0-9]+}} = cir.vaarg %{{[0-9]+}} : (!cir.ptr<[[VALISTTYPE]]>) -> !s32i
    }

    va_end(args);
    // CHECK: cir.vaend %{{[0-9]+}} : !cir.ptr<[[VALISTTYPE]]>

    return count > 0 ? sum / count : 0;
}

int test(void) {
  return average(5, 1, 2, 3, 4, 5);
  // CHECK: cir.call @{{.*}}average{{.*}}(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!s32i, !s32i, !s32i, !s32i, !s32i, !s32i) -> !s32i
}
