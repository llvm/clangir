// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// CXXConstCastExpr l-value emission not implemented
// Location: CIRGenExpr.cpp:2799

int a;
struct b {
  using c = int;
  static c d() { const_cast<int &>(a); }
};
template <typename> struct e {
  static bool f() { b::d; }
};
template <typename... g, typename h> void i(h) { (e<g>::f || ...); }
class j {
  int k;

public:
  void l() { i<int>(k); }
};
void m() {
  j n;
  n.l();
}
