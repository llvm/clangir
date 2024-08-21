// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s 

#include <stdint.h>

void foo(int64_t x) {
  __sync_fetch_and_add(&x, 1);
}

// CHECK: %0 = cir.alloca !s64i, !cir.ptr<!s64i>, ["x", init] {alignment = 8 : i64}