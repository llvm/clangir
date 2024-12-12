//===--- CIRGenDeclCXX.cpp - Build CIR Code for C++ declarations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ declarations
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

void CIRGenModule::emitCXXGlobalInitFunc() {
  while (!CXXGlobalInits.empty() && !CXXGlobalInits.back())
    CXXGlobalInits.pop_back();

  if (CXXGlobalInits.empty()) // TODO(cir): &&
                              // PrioritizedCXXGlobalInits.empty())
    return;

  assert(0 && "NYE");
}

void CIRGenModule::emitCXXGlobalVarDeclInitFunc(const VarDecl *D,
                                                cir::GlobalOp Addr,
                                                bool PerformInit) {
  // According to E.2.3.1 in CUDA-7.5 Programming guide: __device__,
  // __constant__ and __shared__ variables defined in namespace scope,
  // that are of class type, cannot have a non-empty constructor. All
  // the checks have been done in Sema by now. Whatever initializers
  // are allowed are empty and we just need to ignore them here.
  if (getLangOpts().CUDAIsDevice && !getLangOpts().GPUAllowDeviceInit &&
      (D->hasAttr<CUDADeviceAttr>() || D->hasAttr<CUDAConstantAttr>() ||
       D->hasAttr<CUDASharedAttr>()))
    return;

  // Check if we've already initialized this decl.
  auto I = DelayedCXXInitPosition.find(D);
  if (I != DelayedCXXInitPosition.end() && I->second == ~0U)
    return;

  emitCXXGlobalVarDeclInit(D, Addr, PerformInit);
}

void CIRGenFunction::emitCXXGuardedInit(const VarDecl &varDecl,
                                        cir::GlobalOp globalOp,
                                        bool performInit) {
  // If we've been asked to forbid guard variables, emit an error now. This
  // diagnostic is hard-coded for Darwin's use case; we can find better phrasing
  // if someone else needs it.
  if (CGM.getCodeGenOpts().ForbidGuardVariables)
    llvm_unreachable("NYI");

  CGM.getCXXABI().emitGuardedInit(*this, varDecl, globalOp, performInit);
}

void CIRGenFunction::emitCXXGlobalVarDeclInit(const VarDecl &varDecl,
                                              cir::GlobalOp globalOp,
                                              bool performInit) {
  // TODO(CIR): We diverge from CodeGen here via having this in CIRGenModule
  // instead. This is necessary due to the way we are constructing global inits
  // at the moment. With LoweringPrepare being moved to CIRGen we should
  // refactor this to live here.
  llvm_unreachable("NYI");
}
