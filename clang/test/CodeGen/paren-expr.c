// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

int f(int a, int b, int c) {
  // CHECK: add
  return a + (b + c);
}
