// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *

// Test Non-ODR-Use constant DeclRefExpr as lvalue
// This triggers NOUR_Constant handling which is currently NYI

constexpr int global_const = 42;

void test_nour_constant() {
  // Taking address of constexpr - this is a non-ODR-use constant
  const int *p = &global_const;
}
