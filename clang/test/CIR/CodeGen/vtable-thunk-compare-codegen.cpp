// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.codegen.ll
// RUN: FileCheck --input-file=%t.cir.ll %s --check-prefix=CIR
// RUN: FileCheck --input-file=%t.codegen.ll %s --check-prefix=CODEGEN

// Test that CIR thunk generation matches CodeGen behavior

class Base1 {
public:
  virtual void foo() {}
};

class Base2 {
public:
  virtual void bar() {}
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

// Both should have the thunk in the vtable
// CIR-DAG: @_ZTV7Derived = linkonce_odr global{{.*}}@_ZThn{{[0-9]+}}_N7Derived3barEv
// CODEGEN-DAG: @_ZTV7Derived = linkonce_odr {{.*}}constant{{.*}}@_ZThn{{[0-9]+}}_N7Derived3barEv

// Both should generate a thunk
// CIR-DAG: define linkonce_odr void @_ZThn{{[0-9]+}}_N7Derived3barEv
// CODEGEN-DAG: define linkonce_odr void @_ZThn{{[0-9]+}}_N7Derived3barEv
