// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Member function pointer emission NYI
// Location: CIRGenExprConst.cpp:2045

class a {
public:
  int b(unsigned);
};
class c : a {
  struct d {
    int (c::*e)(unsigned);
  } static const f[];
};
const c::d c::f[]{&a::b};
