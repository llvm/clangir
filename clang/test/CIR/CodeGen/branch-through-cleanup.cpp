// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll --check-prefix=OGCG %s

// Test branch-through cleanup destination creation for nested scopes with
// fall-through that is a branch-through, exercising the code path in
// PopCleanupBlock that creates branch-through destinations for enclosing
// cleanup scopes.

struct RAII {
  RAII();
  ~RAII();
  RAII& operator++();
  int& operator*();
  bool operator!=(const RAII&);
};

struct Container {
  RAII begin();
  RAII end();
};

// Test range-for with RAII iterator that creates nested cleanup scopes
// with branch-through semantics when the loop variable needs cleanup.
// CIR-LABEL: @_Z27test_range_for_with_cleanupv
// LLVM-LABEL: @_Z27test_range_for_with_cleanupv
// OGCG-LABEL: @_Z27test_range_for_with_cleanupv
void test_range_for_with_cleanup() {
  Container c;

  // The range-for creates multiple cleanup scopes:
  // 1. Outer scope for range expression
  // 2. Loop body scope for loop variable
  // These scopes have fall-through that is a branch-through when
  // transitioning between iterations, exercising the branch-through
  // destination creation added in this patch.

  // CIR: cir.scope {
  // CIR:   cir.scope {
  // CIR:     cir.call @_ZN9Container5beginEv
  // CIR:   }
  // CIR:   cir.scope {
  // CIR:     cir.call @_ZN9Container3endEv
  // CIR:   }
  // CIR:   cir.for : cond {
  // CIR:     cir.call @_ZN4RAIIneERKS_
  // CIR:   } body {
  // CIR:     cir.call @_ZN4RAIIdeEv
  // CIR:   } step {
  // CIR:     cir.call @_ZN4RAIIppEv
  // CIR:   }
  // CIR:   cir.call @_ZN4RAIID1Ev
  // CIR:   cir.call @_ZN4RAIID1Ev
  // CIR: }

  // Verify LLVM lowering has proper cleanup structure
  // LLVM: %{{.*}} = call %struct.RAII @_ZN9Container5beginEv(
  // LLVM: %{{.*}} = call %struct.RAII @_ZN9Container3endEv(
  // LLVM: %{{.*}} = call i1 @_ZN4RAIIneERKS_(
  // LLVM: %{{.*}} = call ptr @_ZN4RAIIdeEv(
  // LLVM: %{{.*}} = call ptr @_ZN4RAIIppEv(
  // LLVM: call void @_ZN4RAIID1Ev(
  // LLVM: call void @_ZN4RAIID1Ev(

  // Verify OGCG has similar structure (validating our lowering is correct)
  // OGCG: call void @_ZN9Container5beginEv(ptr {{.*}}sret(%struct.RAII)
  // OGCG: call void @_ZN9Container3endEv(ptr {{.*}}sret(%struct.RAII)
  // OGCG: br label %for.cond
  // OGCG: call {{.*}}@_ZN4RAIIneERKS_(
  // OGCG: br i1 %{{.*}}, label %for.body, label %for.cond.cleanup
  // OGCG: call void @_ZN4RAIID1Ev(
  // OGCG: call void @_ZN4RAIID1Ev(
  // OGCG: call {{.*}}@_ZN4RAIIdeEv(
  // OGCG: call {{.*}}@_ZN4RAIIppEv(

  for (auto& elem : c) {
    int val = elem;
    (void)val;
  }
}

// Test nested scopes with temporaries requiring cleanup.
// This creates a scenario where fall-through from the inner scope
// branches through the outer cleanup scope.
// CIR-LABEL: @_Z35test_nested_scopes_with_temporariesv
// LLVM-LABEL: @_Z35test_nested_scopes_with_temporariesv
// OGCG-LABEL: @_Z35test_nested_scopes_with_temporariesv
void test_nested_scopes_with_temporaries() {
  // CIR: cir.scope {
  // CIR:   cir.call @_ZN4RAIIC1Ev
  // CIR:   cir.scope {
  // CIR:     cir.call @_ZN4RAIIC1Ev
  // CIR:     cir.call @_ZN4RAIID1Ev
  // CIR:   }
  // CIR:   cir.call @_ZN4RAIID1Ev
  // CIR: }

  // LLVM: call void @_ZN4RAIIC1Ev(
  // LLVM: call void @_ZN4RAIIC1Ev(
  // LLVM: call void @_ZN4RAIID1Ev(
  // LLVM: call void @_ZN4RAIID1Ev(

  // OGCG: call void @_ZN4RAIIC1Ev(ptr {{.*}}%outer)
  // OGCG: call void @_ZN4RAIIC1Ev(ptr {{.*}}%inner)
  // OGCG: call void @_ZN4RAIID1Ev(ptr {{.*}}%inner)
  // OGCG: call void @_ZN4RAIID1Ev(ptr {{.*}}%outer)

  {
    RAII outer;
    {
      RAII inner;
      // Fall-through from inner scope branches through outer cleanup
    }
  }
}

// Test multiple cleanup scopes in sequence to verify cleanup ordering
// CIR-LABEL: @_Z24test_sequential_cleanupsv
// LLVM-LABEL: @_Z24test_sequential_cleanupsv
// OGCG-LABEL: @_Z24test_sequential_cleanupsv
void test_sequential_cleanups() {
  // CIR: cir.call @_ZN4RAIIC1Ev
  // CIR: cir.scope {
  // CIR:   cir.call @_ZN4RAIIC1Ev
  // CIR:   cir.call @_ZN4RAIID1Ev
  // CIR: }
  // CIR: cir.call @_ZN4RAIIC1Ev
  // CIR: cir.call @_ZN4RAIID1Ev
  // CIR: cir.call @_ZN4RAIID1Ev

  // Verify destruction order in LLVM lowering
  // LLVM: call void @_ZN4RAIIC1Ev(
  // LLVM: call void @_ZN4RAIIC1Ev(
  // LLVM: call void @_ZN4RAIID1Ev(
  // LLVM: call void @_ZN4RAIIC1Ev(
  // LLVM: call void @_ZN4RAIID1Ev(
  // LLVM: call void @_ZN4RAIID1Ev(

  // Verify OGCG has same destruction order
  // OGCG: call void @_ZN4RAIIC1Ev(ptr {{.*}}%r1)
  // OGCG: call void @_ZN4RAIIC1Ev(ptr {{.*}}%r2)
  // OGCG: call void @_ZN4RAIID1Ev(ptr {{.*}}%r2)
  // OGCG: call void @_ZN4RAIIC1Ev(ptr {{.*}}%r3)
  // OGCG: call void @_ZN4RAIID1Ev(ptr {{.*}}%r3)
  // OGCG: call void @_ZN4RAIID1Ev(ptr {{.*}}%r1)

  RAII r1;
  {
    RAII r2;
  }
  RAII r3;
}
