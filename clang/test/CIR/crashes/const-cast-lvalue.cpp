// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Test for CXXConstCastExpr l-value emission not implemented
// Assertion at CIRGenExpr.cpp:2764
//
// This test triggers the assertion:
// "Use emitCastLValue below, remove me when adding testcase"
//
// Original failure: assertion_emitcastlvalue from LLVM build
// Reduced from /tmp/DeltaTree-af9b43.cpp

int a;
struct b {
  using c = int;
  static c d() { const_cast<int &>(a); }
};
struct e {
  static bool f() { b::d; }
};
struct g {
  static bool h() { e::f; }
};
struct j : g {
  using i = int;
  static i k() { h; }
};
void l() { j::k; }
