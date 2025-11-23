// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// Test that functions falling off the end (missing return) get proper terminators.

inline namespace a {
class b {
public:
  ~b();
};
} // namespace a
b c() {
  b d;
  if (0)
    return d;
  // Falls off the end - should emit cir.unreachable
}
