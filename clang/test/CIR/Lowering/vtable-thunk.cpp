// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Test that thunks lower correctly from CIR to LLVM IR

class Base1 {
public:
  virtual void foo() {}
  int x;
};

class Base2 {
public:
  virtual void bar() {}
  int y;
};

class Derived : public Base1, public Base2 {
public:
  void bar() override {}
};

void test() {
  Derived d;
  Base2* b2 = &d;
  b2->bar();
}

// Check vtable contains thunk with correct offset (16 bytes on x86_64)
// LLVM: @_ZTV7Derived = linkonce_odr global
// LLVM: @_ZThn16_N7Derived3barEv

// Check thunk function is defined with correct linkage
// LLVM: define linkonce_odr void @_ZThn16_N7Derived3barEv(ptr{{.*}} %0)

