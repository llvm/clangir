// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

struct s {
  int a;
  float b;
};
int main() { s v; }
// CHECK: memref<!named_tuple.named_tuple<"s", [i32, f32]>>
