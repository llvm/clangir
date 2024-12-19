// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s
// -o %t.cir XFAIL: *

struct E {};
E e;

void throws() { throw e; }

void bar() {
  try {
    throws();
  } catch (E e) {
  }
}
