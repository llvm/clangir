// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o - 2>&1 | FileCheck %s

void reject() {
  for (int i = 0; i < 100; i++, i++);
  // CHECK: cannot handle non-constant step for induction variable
  // CHECK: cannot handle non-canonicalized loop
}
