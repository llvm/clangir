// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int test(int x) {
  static int arr[10] = {0, 1, 0, 0};
  return arr[x];
}
// LLVM: @test.arr = internal global <{ i32, i32, [8 x i32] }> <{ i32 0, i32 1, [8 x i32] zeroinitializer }>