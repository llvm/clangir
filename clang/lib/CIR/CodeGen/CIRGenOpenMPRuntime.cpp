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

CIRGenOpenMPRuntime::CIRGenOpenMPRuntime(CIRGenModule &CGM) : CGM(CGM) {}

Address CIRGenOpenMPRuntime::getAddressOfLocalVariable(CIRGenFunction &CGF,
                                                       const VarDecl *VD) {
  // TODO[OpenMP]: Implement this method.
  return Address::invalid();
}

void CIRGenOpenMPRuntime::checkAndEmitLastprivateConditional(
    CIRGenFunction &CGF, const Expr *LHS) {
  // TODO[OpenMP]: Implement this method.
  return;
}

void CIRGenOpenMPRuntime::registerTargetGlobalVariable(
    const clang::VarDecl *VD, mlir::cir::GlobalOp globalOp) {
  // TODO[OpenMP]: Implement this method.
  return;
}

void CIRGenOpenMPRuntime::emitDeferredTargetDecls() const {
  // TODO[OpenMP]: Implement this method.
  return;
}

void CIRGenOpenMPRuntime::emitFunctionProlog(CIRGenFunction &CGF,
                                             const clang::Decl *D) {
  // TODO[OpenMP]: Implement this method.
  return;
}

bool CIRGenOpenMPRuntime::emitTargetGlobal(clang::GlobalDecl &GD) {
  // TODO[OpenMP]: Implement this method.
  return false;
}
