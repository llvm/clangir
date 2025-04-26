// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

void test_delete_array(int *ptr) {
  delete[] ptr;
}

// CHECK: [[PTR:%[0-9]+]] = load ptr, ptr %{{[0-9]+}}, align 8
// CHECK-NEXT: call void @_ZdaPv(ptr [[PTR]])
