// RUN: %clang_cc1 -triple arm64-apple-darwin -std=c++20 -O2 \
// RUN:   -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s --allow-empty
// CHECK-NOT: NYI
// CHECK-NOT: error

namespace B {
template <class _0p> class B {
public:
  typedef _0p A;
  B() { __has_trivial_destructor(A); }
};
template <class _0p, class _0e0uence = B<_0p>> class A { _0e0uence A; };
} // namespace B
class A { B::A<A> A; } A;

