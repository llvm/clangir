// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check -verify-diagnostics -o %t-out.cir

int *basic() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42;  // expected-warning {{Found invalid use of pointer 'p'}}
  return p; // expected-warning {{Found invalid use of pointer 'p'}}
}