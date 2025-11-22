// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Test basic thunk generation for multiple inheritance with non-virtual thunks

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

// Check that thunk function is generated with correct mangling
// CIR: cir.func linkonce_odr @_ZThn8_N7Derived3barEv
// CIR-SAME: (!cir.ptr<![[DerivedTy:.*]]>)

// Check thunk is in vtable
// CIR: cir.global linkonce_odr @_ZTV7Derived = #cir.vtable
// CIR-SAME: #cir.global_view<@_ZThn8_N7Derived3barEv>

// LLVM: define linkonce_odr void @_ZThn8_N7Derived3barEv
// LLVM-SAME: ptr

// LLVM: @_ZTV7Derived = linkonce_odr global
// LLVM-SAME: @_ZThn8_N7Derived3barEv
