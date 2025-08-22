// RUN: %clang_cc1 -triple arm64-apple-darwin -std=c++20 -O2 -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple arm64-apple-darwin -std=c++20 -O2 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple arm64-apple-darwin -std=c++20 -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

// CIR: cir.const #true

// LLVM-DAG: @A = local_unnamed_addr global %class.A zeroinitializer, align 1
// LLVM-DAG: type { i8 }

namespace B {
template <class _0p> class B {
public:
  typedef _0p A;
  B() { __has_trivial_destructor(A); }
};
template <class _0p, class _0e0uence = B<_0p>> class A { _0e0uence A; };
} // namespace B
class A { B::A<A> A; } A;

