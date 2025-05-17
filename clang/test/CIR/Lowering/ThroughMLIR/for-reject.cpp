// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
// XFAIL: *

void f();

void reject() {
  for (int i = 0; i < 100; i++, f());
  for (int i = 0; i < 100; i++, i++);
}
