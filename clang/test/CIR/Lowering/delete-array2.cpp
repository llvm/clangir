// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm -O0 %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

int *newmem();
struct cls {
  ~cls();
};
cls::~cls() { delete[] newmem(); }

// CHECK: [[NEWMEM:%[0-9]+]] = call ptr @_Z6newmemv()
// CHECK-NEXT: call void @_ZdaPv(ptr [[NEWMEM]])