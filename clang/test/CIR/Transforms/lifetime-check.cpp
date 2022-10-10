// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check="history=invalid,null" -verify-diagnostics -o %t-out.cir

int *p0() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  return p;
}

int *p1(bool b = true) {
  int *p = nullptr; // expected-note {{invalidated here}}
  if (b) {
    int x = 0;
    p = &x;
    *p = 42;
  }        // expected-note {{pointee 'x' invalidated at end of scope}}
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  return p;
}

int *p2() {
  int *p = nullptr; // expected-note {{invalidated here}}
  *p = 42;          // expected-warning {{use of invalid pointer 'p'}}
  return p;
}

int *p3() {
  int *p;
  p = nullptr; // expected-note {{invalidated here}}
  *p = 42;     // expected-warning {{use of invalid pointer 'p'}}
  return p;
}