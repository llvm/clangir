// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -emit-cir -O2 %s -o - | FileCheck %s

// Test case for issue #1938: ensure that alias replacements work correctly
// when operations are not yet in a block during applyReplacements().

class Base {
public:
    Base() {}
    virtual ~Base() {}
};

class Derived : public Base {
public:
    Derived() : Base() {}
    ~Derived() {}
};

void test() {
    Derived d;
}

// CHECK: cir.func @_ZN7DerivedC2Ev
// CHECK: cir.func @_ZN7DerivedD2Ev
// This test primarily checks that compilation succeeds without assertion failures
