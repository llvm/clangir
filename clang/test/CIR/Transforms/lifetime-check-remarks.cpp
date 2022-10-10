// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o %t.cir
// RUN: cir-tool %t.cir -cir-lifetime-check="remarks=pset" -verify-diagnostics -o %t-out.cir

int *p0() {
  int *p = nullptr;
  {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
  return p;
}

int *p1(bool b = true) {
  int *p = nullptr;
  if (b) {
    int x = 0;
    p = &x;
    *p = 42;
  }
  *p = 42; // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid, nullptr }}}
  return p;
}