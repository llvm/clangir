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

#include "Address.h"
#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenTypes.h"
#include "TargetInfo.h"
#include "mlir-c/IR.h"
#include "clang/AST/Attr.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/Specifiers.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace mlir::cir;
using namespace cir;

void CIRGenModule::buildCXXGlobalInitFunc() {
  while (!CXXGlobalInits.empty() && !CXXGlobalInits.back())
    CXXGlobalInits.pop_back();

  if (CXXGlobalInits.empty()) // TODO(cir): &&
                              // PrioritizedCXXGlobalInits.empty())
    return;

  assert(0 && "NYE");
}

void CIRGenModule::buildCXXGlobalVarDeclInitFunc(const VarDecl *D,
                                                 mlir::cir::GlobalOp Addr,
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

  mlir::FunctionType fTy =
      mlir::FunctionType::get(builder.getContext(), VoidTy, VoidTy);
  SmallString<256> fnName;
  {
    llvm::raw_svector_ostream out(fnName);
    getCXXABI().getMangleContext().mangleDynamicInitializer(D, out);
  }

  // Create a variable initialization function.
  mlir::cir::FuncOp fn = createGlobalInitOrCleanUpFunction(
      fTy, fnName.str(), getTypes().arrangeNullaryFunction(), D->getLocation());

  auto *isa = D->getAttr<InitSegAttr>();
  CIRGenFunction(*this, getBuilder())
      .generateCXXGlobalVarDeclInitFunc(fn, D, Addr, PerformInit);

  mlir::cir::GlobalOp comdatKey =
      supportsCOMDAT() && D->isExternallyVisible() ? Addr : nullptr;

  if (D->getTLSKind()) {
    llvm_unreachable("NYI");
  } else if (PerformInit && isa) {
    llvm_unreachable("performinit && isa");
  } else if (auto *ipa = D->getAttr<InitPriorityAttr>()) {
    llvm_unreachable("NYI");
  } else if (isTemplateInstantiation(D->getTemplateSpecializationKind()) ||
             getASTContext().GetGVALinkageForVariable(D) ==
                 clang::GVA_DiscardableODR ||
             D->hasAttr<SelectAnyAttr>()) {
    llvm_unreachable("NYI");
  } else {
    llvm_unreachable("NYI");
  }

  DelayedCXXInitPosition[D] = ~0U;
}

void CIRGenModule::buildCXXGlobalVarDeclInit(const VarDecl *D,
                                             mlir::cir::GlobalOp Addr,
                                             bool PerformInit) {
  QualType T = D->getType();

  // TODO: handle address space
  // The address space of a static local variable (DeclPtr) may be different
  // from the address space of the "this" argument of the constructor. In that
  // case, we need an addrspacecast before calling the constructor.
  //
  // struct StructWithCtor {
  //   __device__ StructWithCtor() {...}
  // };
  // __device__ void foo() {
  //   __shared__ StructWithCtor s;
  //   ...
  // }
  //
  // For example, in the above CUDA code, the static local variable s has a
  // "shared" address space qualifier, but the constructor of StructWithCtor
  // expects "this" in the "generic" address space.
  assert(!MissingFeatures::addressSpace());

  if (!T->isReferenceType()) {
    if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd &&
        D->hasAttr<OMPThreadPrivateDeclAttr>()) {
      llvm_unreachable("NYI");
    }
    bool NeedsDtor =
        D->needsDestruction(getASTContext()) == QualType::DK_cxx_destructor;
    // PerformInit, constant store invariant / destroy handled below.
    bool isCstStorage =
        D->getType().isConstantStorage(getASTContext(), true, !NeedsDtor);
    codegenGlobalInitCxxStructor(D, Addr, PerformInit, NeedsDtor, isCstStorage);
    return;
  }

  assert(PerformInit && "cannot have constant initializer which needs "
                        "destruction for reference");
  // TODO(cir): buildReferenceBindingToExpr
}

mlir::cir::FuncOp CIRGenModule::createGlobalInitOrCleanUpFunction(
    mlir::FunctionType ty, const Twine &name, const CIRGenFunctionInfo &fi,
    SourceLocation loc, bool tls, mlir::cir::GlobalLinkageKind linkage) {
  llvm_unreachable("NYI");
}

/// Emit the code necessary to initialize the given global variable.
void CIRGenFunction::generateCXXGlobalVarDeclInitFunc(mlir::cir::FuncOp fn,
                                                      const VarDecl *varDecl,
                                                      mlir::cir::GlobalOp addr,
                                                      bool performInit) {
  // Check if we need to emit debug info for variable initializer.
  if (varDecl->hasAttr<NoDebugAttr>())
    debugInfo = nullptr;

  curEHLocation = varDecl->getBeginLoc();

  StartFunction(GlobalDecl(varDecl, DynamicInitKind::Initializer),
                CGM.getASTContext().VoidTy, fn,
                getTypes().arrangeNullaryFunction(), FunctionArgList(),
                varDecl->getLocation(), varDecl->getLocation());
  // Emit an artificial location for this function.
  assert(MissingFeatures::generateDebugInfo());

  // Use guarded initialization if the global variable is weak. This occurs for,
  // e.g., instantiated static data members and definitions explicitly marked
  // weak.
  //
  // Also use guarded initialization for a variable with dynamic TLS and
  // unordered initialization. (If the initialization is ordered, the ABI layer
  // will guard the whole-TU initialization for us.)
  if (addr.hasExternalWeakLinkage() || addr.hasLinkOnceLinkage() ||
      (varDecl->getTLSKind() == VarDecl::TLS_Dynamic &&
       isTemplateInstantiation(varDecl->getTemplateSpecializationKind()))) {
    llvm_unreachable("NYI");
  } else {
    llvm_unreachable("NYI");
  }

  if (getLangOpts().HLSL)
    llvm_unreachable("NYI");

  finishFunction(varDecl->getLocation());
}
