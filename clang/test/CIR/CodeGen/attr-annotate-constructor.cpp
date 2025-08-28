// RUN: %clang_cc1 -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file %t.cir %s

class Foo {
public:
  [[clang::annotate("test")]] Foo() {}
};
// CHECK: #cir.annotation<name = "test", args = []>

Foo foo;
