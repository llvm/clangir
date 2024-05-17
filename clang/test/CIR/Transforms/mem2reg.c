// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: cir.func @foo
    int foo(int* ar, int n) {
  int sum = 0;
  for (int i = 0; i < n; ++i)
    sum += ar[i];
  return sum;
}