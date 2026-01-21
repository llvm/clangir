// RUN: %clang_cc1 -O2 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -O2 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

__attribute__((hot)) int s0(int a, int b) {
  int x = a + b;
  return x;
}

// CIR:      cir.func {{.*}} @_Z2s0ii(
// CIR-SAME:     extra(#cir<extra({{{.*}}hot = #cir.hot

// LLVM: define dso_local i32 @_Z2s0ii({{.*}} #[[#ATTR1:]] {
// LLVM: attributes #[[#ATTR1]] = {{.*}}hot

