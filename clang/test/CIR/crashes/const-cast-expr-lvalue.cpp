// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// CXXConstCastExpr l-value emission not implemented
// Location: CIRGenExpr.cpp:2720

void f() {
  const int x = 0;
  const_cast<int&>(x) = 1;
}
