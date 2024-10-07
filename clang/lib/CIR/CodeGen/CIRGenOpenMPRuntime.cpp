//===--- CIRGenStmtOpenMP.cpp - Interface to OpenMP Runtimes --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime MLIR code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenMPRuntime.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

using namespace cir;
using namespace clang;

CIRGenOpenMPRuntime::CIRGenOpenMPRuntime(CIRGenModule &cgm) : CGM(cgm) {}

Address CIRGenOpenMPRuntime::getAddressOfLocalVariable(CIRGenFunction &cgf,
                                                       const VarDecl *vd) {
  assert(!MissingFeatures::openMPRuntime());
  return Address::invalid();
}

void CIRGenOpenMPRuntime::checkAndEmitLastprivateConditional(
    CIRGenFunction &cgf, const Expr *lhs) {
  assert(!MissingFeatures::openMPRuntime());
}

void CIRGenOpenMPRuntime::registerTargetGlobalVariable(
    const clang::VarDecl *vd, mlir::cir::GlobalOp globalOp) {
  assert(!MissingFeatures::openMPRuntime());
}

void CIRGenOpenMPRuntime::emitDeferredTargetDecls() const {
  assert(!MissingFeatures::openMPRuntime());
}

void CIRGenOpenMPRuntime::emitFunctionProlog(CIRGenFunction &cgf,
                                             const clang::Decl *d) {
  assert(!MissingFeatures::openMPRuntime());
}

bool CIRGenOpenMPRuntime::emitTargetGlobal(clang::GlobalDecl &gd) {
  assert(!MissingFeatures::openMPRuntime());
  return false;
}

void CIRGenOpenMPRuntime::emitTaskWaitCall(CIRGenBuilderTy &builder,
                                           CIRGenFunction &cgf,
                                           mlir::Location loc,
                                           const OMPTaskDataTy &data) {

  if (!cgf.HaveInsertPoint())
    return;

  if (cgf.cgm.getLangOpts().OpenMPIRBuilder && data.Dependences.empty()) {
    // TODO: Need to support taskwait with dependences in the OpenMPIRBuilder.
    // TODO(cir): This could change in the near future when OpenMP 5.0 gets
    // supported by MLIR
    llvm_unreachable("NYI");
    // builder.create<mlir::omp::TaskwaitOp>(Loc);
  } else {
    llvm_unreachable("NYI");
  }
  assert(!MissingFeatures::openMPRegionInfo());
}

void CIRGenOpenMPRuntime::emitBarrierCall(CIRGenBuilderTy &builder,
                                          CIRGenFunction &cgf,
                                          mlir::Location loc) {

  assert(!MissingFeatures::openMPRegionInfo());

  if (cgf.cgm.getLangOpts().OpenMPIRBuilder) {
    builder.create<mlir::omp::BarrierOp>(loc);
    return;
  }

  if (!cgf.HaveInsertPoint())
    return;

  llvm_unreachable("NYI");
}

void CIRGenOpenMPRuntime::emitTaskyieldCall(CIRGenBuilderTy &builder,
                                            CIRGenFunction &cgf,
                                            mlir::Location loc) {

  if (!cgf.HaveInsertPoint())
    return;

  if (cgf.cgm.getLangOpts().OpenMPIRBuilder) {
    builder.create<mlir::omp::TaskyieldOp>(loc);
  } else {
    llvm_unreachable("NYI");
  }

  assert(!MissingFeatures::openMPRegionInfo());
}
