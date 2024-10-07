//===----- CirGenCXXABI.cpp - Interface to C++ ABIs -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"

#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"

using namespace cir;
using namespace clang;

CIRGenCXXABI::~CIRGenCXXABI() = default;

CIRGenCXXABI::AddedStructorArgCounts CIRGenCXXABI::addImplicitConstructorArgs(
    CIRGenFunction &cgf, const clang::CXXConstructorDecl *d,
    clang::CXXCtorType type, bool forVirtualBase, bool delegating,
    CallArgList &args) {
  auto addedArgs =
      getImplicitConstructorArgs(cgf, d, type, forVirtualBase, delegating);
  for (size_t i = 0; i < addedArgs.Prefix.size(); ++i)
    args.insert(args.begin() + 1 + i,
                CallArg(RValue::get(addedArgs.Prefix[i].Value),
                        addedArgs.Prefix[i].Type));
  for (const auto &arg : addedArgs.Suffix)
    args.add(RValue::get(arg.Value), arg.Type);
  return AddedStructorArgCounts(addedArgs.Prefix.size(),
                                addedArgs.Suffix.size());
}

CatchTypeInfo CIRGenCXXABI::getCatchAllTypeInfo() {
  return CatchTypeInfo{nullptr, 0};
}

bool CIRGenCXXABI::NeedsVTTParameter(GlobalDecl gd) { return false; }

void CIRGenCXXABI::buildThisParam(CIRGenFunction &cgf,
                                  FunctionArgList &params) {
  const auto *md = cast<CXXMethodDecl>(cgf.CurGD.getDecl());

  // FIXME: I'm not entirely sure I like using a fake decl just for code
  // generation. Maybe we can come up with a better way?
  auto *thisDecl =
      ImplicitParamDecl::Create(CGM.getASTContext(), nullptr, md->getLocation(),
                                &CGM.getASTContext().Idents.get("this"),
                                md->getThisType(), ImplicitParamKind::CXXThis);
  params.push_back(thisDecl);
  cgf.CXXABIThisDecl = thisDecl;

  // Compute the presumed alignment of 'this', which basically comes down to
  // whether we know it's a complete object or not.
  auto &layout = cgf.getContext().getASTRecordLayout(md->getParent());
  if (md->getParent()->getNumVBases() == 0 ||
      md->getParent()->isEffectivelyFinal() ||
      isThisCompleteObject(cgf.CurGD)) {
    cgf.CXXABIThisAlignment = layout.getAlignment();
  } else {
    llvm_unreachable("NYI");
  }
}

mlir::cir::GlobalLinkageKind CIRGenCXXABI::getCXXDestructorLinkage(
    GVALinkage linkage, const CXXDestructorDecl *dtor, CXXDtorType dt) const {
  // Delegate back to CGM by default.
  return CGM.getCIRLinkageForDeclarator(dtor, linkage,
                                        /*IsConstantVariable=*/false);
}