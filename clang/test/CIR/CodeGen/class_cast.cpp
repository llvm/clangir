// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=OGCG
class Base {
  // CIR-LABEL: _ZN4BaseaSERKS_
  // CIR-SAME: [[ARG0:%.*]]: !cir.ptr{{.*}}, [[ARG1:%.*]]: !cir.ptr{{.*}}
  // CIR: [[ALLOCA_0:%.*]] =  cir.alloca
  // CIR: [[ALLOCA_1:%.*]] =  cir.alloca
  // CIR: [[ALLOCA_2:%.*]] =  cir.alloca
  // CIR: cir.store [[ARG0]], [[ALLOCA_0]]
  // CIR: cir.store [[ARG1]], [[ALLOCA_1]]
  // CIR: [[LD_0:%.*]] = cir.load deref [[ALLOCA_0]]
  // CIR: cir.store align(8) [[LD_0]], [[ALLOCA_2]]
  // CIR: [[LD_1:%.*]] = cir.load [[ALLOCA_2]]
  // CIR: cir.return [[LD_1]]
};

class Derived : Base {
  Derived &operator=(Derived &);
};
Derived &Derived::operator=(Derived &B) {
  // CIR-LABEL: _ZN7DerivedaSERS_
  // CIR-SAME: [[ARG0:%.*]]: !cir.ptr{{.*}}, [[ARG1:%.*]]: !cir.ptr{{.*}}
  // CIR: cir.store [[ARG0]], [[ALLOCA_0:%.*]] :
  // CIR: cir.store [[ARG1]], [[ALLOCA_1:%.*]] :
  // CIR: [[LD_0:%.*]] = cir.load [[ALLOCA_0]]
  // CIR: [[BASE_ADDR_0:%.*]] = cir.base_class_addr [[LD_0]]
  // CIR: [[LD_1:%.*]] = cir.load [[ALLOCA_1]]
  // CIR: [[BASE_ADDR_1:%.*]] = cir.base_class_addr [[LD_1]]
  // CIR: [[CALL:%.*]] = cir.call @_ZN4BaseaSERKS_
  // CIR: [[DERIVED_ADDR:%.*]] = cir.derived_class_addr [[CALL]]
  // CIR: cir.store align(8) [[DERIVED_ADDR]], [[ALLOCA_2:%.*]] :
  // CIR: [[LD_2:%.*]] = cir.load [[ALLOCA_2]]
  // CIR: cir.return [[LD_2]]
  //
  // LLVM-LABEL: _ZN7DerivedaSERS_
  // LLVM-SAME: ptr{{.*}}[[ARG0:%.*]], ptr{{.*}}[[ARG1:%.*]]
  // LLVM: ret ptr [[ARG0]]
  //
  // OGCG-LABEL: _ZN7DerivedaSERS_
  // OGCG-SAME: ptr{{.*}}[[ARG0:%.*]], ptr{{.*}}[[ARG1:%.*]]
  // OGCG: ret ptr [[ARG0]]
  return (Derived &)Base::operator=(B);
}
