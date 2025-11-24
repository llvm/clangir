// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// Test branch-through cleanup support for functions with multiple cleanup scopes.
// This test verifies that when returning from a function with local variables
// that have destructors, the cleanup code properly threads through nested scopes.
//
// NOTE: CIR→LLVM lowering currently returns structs by value instead of using
// the sret calling convention. This is a known ABI limitation, not related to
// cleanup logic. This test focuses on verifying that destructors are called in
// the correct order regardless of the calling convention used.
//
// Reduced from MicrosoftDemangleNodes.cpp

class c {
public:
  ~c();
};
struct d {
  template <typename> using ac = c;
};
struct e {
  typedef d::ac<int> ae;
};
class f {
public:
  e::ae ak;
  template <typename g> f(g, g);
};
struct h {
  f i() const;
};
class j {
public:
  ~j();
};

// CIR: cir.func dso_local @_ZNK1h1iEv
f h::i() const {
  // Local variable 'a' requires cleanup (destructor call)
  j a;
  // CIR: cir.alloca !rec_j, !cir.ptr<!rec_j>, ["a"]

  // Temporary 'b' also requires cleanup
  f b(0, 0);
  // CIR: cir.call @_ZN1fC1IiEET_S1_

  // Return value requires proper cleanup threading
  return b;
  // CIR: cir.call @_ZN1jD1Ev
}
// CIR: }

// LLVM lowering - returns by value (not ABI-compliant sret, but cleanup is correct)
// LLVM: define dso_local %class.f @_ZNK1h1iEv(ptr %{{[0-9]+}})
// LLVM:   alloca %class.j
// LLVM:   call void @_ZN1fC1IiEET_S1_(ptr %{{[0-9]+}}, i32 0, i32 0)
// LLVM:   call void @_ZN1jD1Ev(ptr %{{[0-9]+}})

// Original CodeGen - uses sret calling convention (ABI-compliant)
// OGCG: define dso_local void @_ZNK1h1iEv(ptr {{.*}}sret(%class.f){{.*}} %agg.result, ptr {{.*}} %this)
// OGCG:   alloca %class.j
// OGCG:   call void @_ZN1fC1IiEET_S1_(
// OGCG:   call void @_ZN1jD1Ev(
// OGCG:   ret void
