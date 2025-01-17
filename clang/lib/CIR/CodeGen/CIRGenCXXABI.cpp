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

using namespace clang;
using namespace clang::CIRGen;

CIRGenCXXABI::~CIRGenCXXABI() {}

CIRGenCXXABI::AddedStructorArgCounts CIRGenCXXABI::addImplicitConstructorArgs(
    CIRGenFunction &CGF, const clang::CXXConstructorDecl *D,
    clang::CXXCtorType Type, bool ForVirtualBase, bool Delegating,
    CallArgList &Args) {
  auto AddedArgs =
      getImplicitConstructorArgs(CGF, D, Type, ForVirtualBase, Delegating);
  for (size_t i = 0; i < AddedArgs.Prefix.size(); ++i)
    Args.insert(Args.begin() + 1 + i,
                CallArg(RValue::get(AddedArgs.Prefix[i].Value),
                        AddedArgs.Prefix[i].Type));
  for (const auto &arg : AddedArgs.Suffix)
    Args.add(RValue::get(arg.Value), arg.Type);
  return AddedStructorArgCounts(AddedArgs.Prefix.size(),
                                AddedArgs.Suffix.size());
}

CatchTypeInfo CIRGenCXXABI::getCatchAllTypeInfo() {
  return CatchTypeInfo{nullptr, 0};
}

bool CIRGenCXXABI::NeedsVTTParameter(GlobalDecl GD) { return false; }

void CIRGenCXXABI::buildThisParam(CIRGenFunction &CGF,
                                  FunctionArgList &params) {
  const auto *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());

  // FIXME: I'm not entirely sure I like using a fake decl just for code
  // generation. Maybe we can come up with a better way?
  auto *ThisDecl =
      ImplicitParamDecl::Create(CGM.getASTContext(), nullptr, MD->getLocation(),
                                &CGM.getASTContext().Idents.get("this"),
                                MD->getThisType(), ImplicitParamKind::CXXThis);
  params.push_back(ThisDecl);
  CGF.CXXABIThisDecl = ThisDecl;

  // Compute the presumed alignment of 'this', which basically comes down to
  // whether we know it's a complete object or not.
  auto &Layout = CGF.getContext().getASTRecordLayout(MD->getParent());
  if (MD->getParent()->getNumVBases() == 0 ||
      MD->getParent()->isEffectivelyFinal() ||
      isThisCompleteObject(CGF.CurGD)) {
    CGF.CXXABIThisAlignment = Layout.getAlignment();
  } else {
    CGF.CXXABIThisAlignment = Layout.getNonVirtualAlignment();
  }
}

cir::GlobalLinkageKind CIRGenCXXABI::getCXXDestructorLinkage(
    GVALinkage Linkage, const CXXDestructorDecl *Dtor, CXXDtorType DT) const {
  // Delegate back to CGM by default.
  return CGM.getCIRLinkageForDeclarator(Dtor, Linkage,
                                        /*IsConstantVariable=*/false);
}

std::vector<CharUnits> CIRGenCXXABI::getVBPtrOffsets(const CXXRecordDecl *RD) {
  return std::vector<CharUnits>();
}

bool CIRGenCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  // Fake answer.
  return true;
}

void CIRGenCXXABI::errorUnsupportedABI(CIRGenFunction &CGF, StringRef S) {
  DiagnosticsEngine &Diags = CGF.CGM.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                          "cannot yet compile %0 in this ABI");
  Diags.Report(CGF.getContext().getFullLoc(CGF.CurCodeDecl->getLocation()),
               DiagID)
      << S;
}

CharUnits CIRGenCXXABI::getArrayCookieSize(const CXXNewExpr *E) {
  if (!requiresArrayCookie(E))
    return CharUnits::Zero();
  return getArrayCookieSizeImpl(E->getAllocatedType());
}

CharUnits CIRGenCXXABI::getArrayCookieSizeImpl(QualType ElementType) {
  // BOGUS
  return CharUnits::Zero();
}

Address CIRGenCXXABI::initializeArrayCookie(CIRGenFunction &CGF, Address NewPtr,
                                            mlir::Value NumElements,
                                            const CXXNewExpr *E,
                                            QualType ElementType) {
  // Should never be called.
  errorUnsupportedABI(CGF, "array cookie initialization");
  return Address::invalid();
}

bool CIRGenCXXABI::requiresArrayCookie(const CXXDeleteExpr *E,
                                       QualType ElementType) {
  // If the class's usual deallocation function takes two arguments,
  // it needs a cookie.
  if (E->doesUsualArrayDeleteWantSize())
    return true;

  return ElementType.isDestructedType();
}

bool CIRGenCXXABI::requiresArrayCookie(const CXXNewExpr *E) {
  // If the class's usual deallocation function takes two arguments,
  // it needs a cookie.
  if (E->doesUsualArrayDeleteWantSize())
    return true;

  return E->getAllocatedType().isDestructedType();
}

void CIRGenCXXABI::readArrayCookie(CIRGenFunction &CGF, Address Ptr,
                                   const CXXDeleteExpr *E, QualType ElementType,
                                   mlir::Value &NumElements,
                                   mlir::Value &AllocPtr,
                                   CharUnits &CookieSize) {
  // Derive a char* in the same address space as the pointer.
  Ptr = Ptr.withElementType(CGF.UInt8Ty);

  // If we don't need an array cookie, bail out early.
  if (!requiresArrayCookie(E, ElementType)) {
    AllocPtr = Ptr.emitRawPointer();
    NumElements = nullptr;
    CookieSize = CharUnits::Zero();
    return;
  }

  CookieSize = getArrayCookieSizeImpl(ElementType);
  mlir::Location Loc = CGF.getLoc(E->getSourceRange());
  int64_t Offset = -(CookieSize.getQuantity());
  auto CastOp = CGF.getBuilder().createPtrBitcast(
      Ptr.getPointer(), CGF.getBuilder().getUIntNTy(8));
  auto AlignOp = CGF.getBuilder().getSignedInt(Loc, Offset, /*width=*/32);
  AllocPtr = CGF.getBuilder().createPtrStride(Loc, CastOp, AlignOp);
  Address AllocAddr = Address(AllocPtr, Ptr.getType(), Ptr.getAlignment());
  NumElements = readArrayCookieImpl(CGF, AllocAddr, CookieSize);
}

mlir::Value CIRGenCXXABI::readArrayCookieImpl(CIRGenFunction &CGF, Address Ptr,
                                              CharUnits CookieSize) {
  errorUnsupportedABI(CGF, "reading a new[] cookie");
  assert(CGF.currSrcLoc && "expected to inherit some source location");
  mlir::Location Loc = *CGF.currSrcLoc;
  return CGF.getBuilder().getSignedInt(Loc, 0, CGF.getSizeSize().getQuantity());
}
