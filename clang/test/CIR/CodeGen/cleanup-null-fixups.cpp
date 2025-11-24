// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.codegen.ll
// RUN: FileCheck --input-file=%t.codegen.ll %s -check-prefix=CODEGEN

// Test for popNullFixups implementation - ensures proper cleanup handling
// when code has unreachable statements after return. The unreachable statement
// c(0) creates branch fixups with null destinations that need to be popped.

inline namespace a {
class c {
public:
  template <typename b> c(b);
  ~c();
};
} // namespace a
class d {
  c e() const;
};
class aj {
public:
  ~aj();
} an;

// CIR-LABEL: cir.func{{.*}}@_ZNK1d1eEv
// CIR: %[[AO:.*]] = cir.alloca !rec_aj, !cir.ptr<!rec_aj>, ["ao"]
// CIR: cir.scope {
// CIR:   %[[AGG_TMP:.*]] = cir.alloca !rec_aj, !cir.ptr<!rec_aj>, ["agg.tmp0"]
// CIR:   %[[AN:.*]] = cir.get_global @an
// CIR:   cir.copy %[[AN]] to %[[AGG_TMP]]
// CIR:   cir.call @_ZN1a1cC1I2ajEET_
// CIR:   cir.call @_ZN2ajD1Ev(%[[AGG_TMP]])
// CIR: } loc
// CIR: cir.call @_ZN2ajD1Ev(%[[AO]])
// CIR: cir.return

// LLVM-LABEL: define{{.*}}@_ZNK1d1eEv
// LLVM: call void @_ZN1a1cC1I2ajEET_
// LLVM: call void @_ZN2ajD1Ev
// LLVM: call void @_ZN2ajD1Ev
// LLVM: ret

// CODEGEN-LABEL: define{{.*}}@_ZNK1d1eEv
// CODEGEN: call void @_ZN1a1cC1I2ajEET_
// CODEGEN: call void @_ZN2ajD1Ev
// CODEGEN: call void @_ZN2ajD1Ev
// CODEGEN: ret

c d::e() const {
  aj ao;
  return an;
  c(0); // Unreachable - should not be emitted in CIR or LLVM
}
