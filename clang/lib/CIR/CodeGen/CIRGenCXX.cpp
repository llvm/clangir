//===--- CGCXX.cpp - Emit LLVM Code for declarations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace clang;
using namespace cir;

/// Try to emit a base destructor as an alias to its primary
/// base-class destructor.
bool CIRGenModule::tryEmitBaseDestructorAsAlias(const CXXDestructorDecl *d) {
  if (!getCodeGenOpts().CXXCtorDtorAliases)
    return true;

  // Producing an alias to a base class ctor/dtor can degrade debug quality
  // as the debugger cannot tell them apart.
  if (getCodeGenOpts().OptimizationLevel == 0)
    return true;

  // If sanitizing memory to check for use-after-dtor, do not emit as
  //  an alias, unless this class owns no members.
  if (getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
      !d->getParent()->field_empty())
    assert(!MissingFeatures::sanitizeDtor());

  // If the destructor doesn't have a trivial body, we have to emit it
  // separately.
  if (!d->hasTrivialBody())
    return true;

  const CXXRecordDecl *Class = d->getParent();

  // We are going to instrument this destructor, so give up even if it is
  // currently empty.
  if (Class->mayInsertExtraPadding())
    return true;

  // If we need to manipulate a VTT parameter, give up.
  if (Class->getNumVBases()) {
    // Extra Credit:  passing extra parameters is perfectly safe
    // in many calling conventions, so only bail out if the ctor's
    // calling convention is nonstandard.
    return true;
  }

  // If any field has a non-trivial destructor, we have to emit the
  // destructor separately.
  for (const auto *i : Class->fields())
    if (i->getType().isDestructedType())
      return true;

  // Try to find a unique base class with a non-trivial destructor.
  const CXXRecordDecl *uniqueBase = nullptr;
  for (const auto &i : Class->bases()) {

    // We're in the base destructor, so skip virtual bases.
    if (i.isVirtual())
      continue;

    // Skip base classes with trivial destructors.
    const auto *base =
        cast<CXXRecordDecl>(i.getType()->castAs<RecordType>()->getDecl());
    if (base->hasTrivialDestructor())
      continue;

    // If we've already found a base class with a non-trivial
    // destructor, give up.
    if (uniqueBase)
      return true;
    uniqueBase = base;
  }

  // If we didn't find any bases with a non-trivial destructor, then
  // the base destructor is actually effectively trivial, which can
  // happen if it was needlessly user-defined or if there are virtual
  // bases with non-trivial destructors.
  if (!uniqueBase)
    return true;

  // If the base is at a non-zero offset, give up.
  const ASTRecordLayout &classLayout = astCtx.getASTRecordLayout(Class);
  if (!classLayout.getBaseClassOffset(uniqueBase).isZero())
    return true;

  // Give up if the calling conventions don't match. We could update the call,
  // but it is probably not worth it.
  const CXXDestructorDecl *baseD = uniqueBase->getDestructor();
  if (baseD->getType()->castAs<FunctionType>()->getCallConv() !=
      d->getType()->castAs<FunctionType>()->getCallConv())
    return true;

  GlobalDecl aliasDecl(d, Dtor_Base);
  GlobalDecl targetDecl(baseD, Dtor_Base);

  // The alias will use the linkage of the referent.  If we can't
  // support aliases with that linkage, fail.
  auto linkage = getFunctionLinkage(aliasDecl);

  // We can't use an alias if the linkage is not valid for one.
  if (!mlir::cir::isValidLinkage(linkage))
    return true;

  auto targetLinkage = getFunctionLinkage(targetDecl);

  // Check if we have it already.
  StringRef mangledName = getMangledName(aliasDecl);
  auto *entry = getGlobalValue(mangledName);
  auto globalValue = dyn_cast<mlir::cir::CIRGlobalValueInterface>(entry);
  if (entry && globalValue && !globalValue.isDeclaration())
    return false;
  if (Replacements.count(mangledName))
    return false;

  assert(globalValue && "only knows how to handle GlobalValue");
  [[maybe_unused]] auto aliasValueType = getTypes().GetFunctionType(aliasDecl);

  // Find the referent.
  auto aliasee = cast<mlir::cir::FuncOp>(GetAddrOfGlobal(targetDecl));
  auto aliaseeGv = dyn_cast_or_null<mlir::cir::CIRGlobalValueInterface>(
      GetAddrOfGlobal(targetDecl));
  // Instead of creating as alias to a linkonce_odr, replace all of the uses
  // of the aliasee.
  if (mlir::cir::isDiscardableIfUnused(linkage) &&
      (targetLinkage !=
           mlir::cir::GlobalLinkageKind::AvailableExternallyLinkage ||
       !targetDecl.getDecl()->hasAttr<AlwaysInlineAttr>())) {
    // FIXME: An extern template instantiation will create functions with
    // linkage "AvailableExternally". In libc++, some classes also define
    // members with attribute "AlwaysInline" and expect no reference to
    // be generated. It is desirable to reenable this optimisation after
    // corresponding LLVM changes.
    llvm_unreachable("NYI");
  }

  // If we have a weak, non-discardable alias (weak, weak_odr), like an
  // extern template instantiation or a dllexported class, avoid forming it on
  // COFF. A COFF weak external alias cannot satisfy a normal undefined
  // symbol reference from another TU. The other TU must also mark the
  // referenced symbol as weak, which we cannot rely on.
  if (mlir::cir::isWeakForLinker(linkage) && getTriple().isOSBinFormatCOFF()) {
    llvm_unreachable("NYI");
  }

  // If we don't have a definition for the destructor yet or the definition
  // is
  // avaialable_externally, don't emit an alias.  We can't emit aliases to
  // declarations; that's just not how aliases work.
  if (aliaseeGv && aliaseeGv.isDeclarationForLinker())
    return true;

  // Don't create an alias to a linker weak symbol. This avoids producing
  // different COMDATs in different TUs. Another option would be to
  // output the alias both for weak_odr and linkonce_odr, but that
  // requires explicit comdat support in the IL.
  if (mlir::cir::isWeakForLinker(targetLinkage))
    llvm_unreachable("NYI");

  // Create the alias with no name.
  buildAliasForGlobal("", entry, aliasDecl, aliasee, linkage);
  return false;
}

static void buildDeclInit(CIRGenFunction &cgf, const VarDecl *d,
                          Address declPtr) {
  assert((d->hasGlobalStorage() ||
          (d->hasLocalStorage() &&
           cgf.getContext().getLangOpts().OpenCLCPlusPlus)) &&
         "VarDecl must have global or local (in the case of OpenCL) storage!");
  assert(!d->getType()->isReferenceType() &&
         "Should not call buildDeclInit on a reference!");

  QualType type = d->getType();
  LValue lv = cgf.makeAddrLValue(declPtr, type);

  const Expr *init = d->getInit();
  switch (CIRGenFunction::getEvaluationKind(type)) {
  case TEK_Aggregate:
    cgf.buildAggExpr(
        init, AggValueSlot::forLValue(lv, AggValueSlot::IsDestructed,
                                      AggValueSlot::DoesNotNeedGCBarriers,
                                      AggValueSlot::IsNotAliased,
                                      AggValueSlot::DoesNotOverlap));
    return;
  case TEK_Scalar:
    cgf.buildScalarInit(init, cgf.getLoc(d->getLocation()), lv, false);
    return;
  case TEK_Complex:
    llvm_unreachable("complext evaluation NYI");
  }
}

static void buildDeclDestroy(CIRGenFunction &cgf, const VarDecl *d) {
  // Honor __attribute__((no_destroy)) and bail instead of attempting
  // to emit a reference to a possibly nonexistent destructor, which
  // in turn can cause a crash. This will result in a global constructor
  // that isn't balanced out by a destructor call as intended by the
  // attribute. This also checks for -fno-c++-static-destructors and
  // bails even if the attribute is not present.
  QualType::DestructionKind dtorKind = d->needsDestruction(cgf.getContext());

  // FIXME:  __attribute__((cleanup)) ?

  switch (dtorKind) {
  case QualType::DK_none:
    return;

  case QualType::DK_cxx_destructor:
    break;

  case QualType::DK_objc_strong_lifetime:
  case QualType::DK_objc_weak_lifetime:
  case QualType::DK_nontrivial_c_struct:
    // We don't care about releasing objects during process teardown.
    assert(!d->getTLSKind() && "should have rejected this");
    return;
  }

  auto &cgm = cgf.cgm;
  QualType type = d->getType();

  // Special-case non-array C++ destructors, if they have the right signature.
  // Under some ABIs, destructors return this instead of void, and cannot be
  // passed directly to __cxa_atexit if the target does not allow this
  // mismatch.
  const CXXRecordDecl *record = type->getAsCXXRecordDecl();
  bool canRegisterDestructor =
      record && (!cgm.getCXXABI().HasThisReturn(
                     GlobalDecl(record->getDestructor(), Dtor_Complete)) ||
                 cgm.getCXXABI().canCallMismatchedFunctionType());

  // If __cxa_atexit is disabled via a flag, a different helper function is
  // generated elsewhere which uses atexit instead, and it takes the destructor
  // directly.
  auto usingExternalHelper = cgm.getCodeGenOpts().CXAAtExit;
  mlir::cir::FuncOp fnOp;
  if (record && (canRegisterDestructor || usingExternalHelper)) {
    assert(!d->getTLSKind() && "TLS NYI");
    assert(!record->hasTrivialDestructor());
    assert(!MissingFeatures::openCLCXX());
    CXXDestructorDecl *dtor = record->getDestructor();
    // In LLVM OG codegen this is done in registerGlobalDtor, but CIRGen
    // relies on LoweringPrepare for further decoupling, so build the
    // call right here.
    auto gd = GlobalDecl(dtor, Dtor_Complete);
    auto structorInfo = cgm.getAddrAndTypeOfCXXStructor(gd);
    fnOp = structorInfo.second;
    cgf.getBuilder().createCallOp(
        cgf.getLoc(d->getSourceRange()),
        mlir::FlatSymbolRefAttr::get(fnOp.getSymNameAttr()),
        mlir::ValueRange{cgf.cgm.getAddrOfGlobalVar(d)});
  } else {
    llvm_unreachable("array destructors not yet supported!");
  }
  assert(fnOp && "expected cir.func");
  cgm.getCXXABI().registerGlobalDtor(cgf, d, fnOp, nullptr);
}

mlir::cir::FuncOp CIRGenModule::codegenCXXStructor(GlobalDecl gd) {
  const auto &fnInfo = getTypes().arrangeCXXStructorDeclaration(gd);
  auto fn = getAddrOfCXXStructor(gd, &fnInfo, /*FnType=*/nullptr,
                                 /*DontDefer=*/true, ForDefinition);

  setFunctionLinkage(gd, fn);
  CIRGenFunction cgf{*this, builder};
  CurCGF = &cgf;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.generateCode(gd, fn, fnInfo);
  }
  CurCGF = nullptr;

  setNonAliasAttributes(gd, fn);
  setCIRFunctionAttributesForDefinition(cast<CXXMethodDecl>(gd.getDecl()), fn);
  return fn;
}

/// Emit code to cause the variable at the given address to be considered as
/// constant from this point onwards.
static void buildDeclInvariant(CIRGenFunction &cgf, const VarDecl *d) {
  return cgf.buildInvariantStart(
      cgf.getContext().getTypeSizeInChars(d->getType()));
}

void CIRGenFunction::buildInvariantStart([[maybe_unused]] CharUnits size) {
  // Do not emit the intrinsic if we're not optimizing.
  if (!cgm.getCodeGenOpts().OptimizationLevel)
    return;

  assert(!MissingFeatures::createInvariantIntrinsic());
}

void CIRGenModule::codegenGlobalInitCxxStructor(const VarDecl *d,
                                                mlir::cir::GlobalOp addr,
                                                bool needsCtor, bool needsDtor,
                                                bool isCstStorage) {
  assert(d && " Expected a global declaration!");
  CIRGenFunction cgf{*this, builder, true};
  CurCGF = &cgf;
  CurCGF->CurFn = addr;
  addr.setAstAttr(mlir::cir::ASTVarDeclAttr::get(builder.getContext(), d));

  if (needsCtor) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *block = builder.createBlock(&addr.getCtorRegion());
    CIRGenFunction::LexicalScope lexScope{*CurCGF, addr.getLoc(),
                                          builder.getInsertionBlock()};
    lexScope.setAsGlobalInit();

    builder.setInsertionPointToStart(block);
    Address declAddr(getAddrOfGlobalVar(d), getASTContext().getDeclAlign(d));
    buildDeclInit(cgf, d, declAddr);
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::cir::YieldOp>(addr->getLoc());
  }

  if (isCstStorage) {
    // TODO: this leads to a missing feature in the moment, probably also need a
    // LexicalScope to be inserted here.
    buildDeclInvariant(cgf, d);
  } else {
    // If not constant storage we'll emit this regardless of NeedsDtor value.
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto *block = builder.createBlock(&addr.getDtorRegion());
    CIRGenFunction::LexicalScope lexScope{*CurCGF, addr.getLoc(),
                                          builder.getInsertionBlock()};
    lexScope.setAsGlobalInit();

    builder.setInsertionPointToStart(block);
    buildDeclDestroy(cgf, d);
    builder.setInsertionPointToEnd(block);
    if (block->empty()) {
      block->erase();
      // Don't confuse lexical cleanup.
      builder.clearInsertionPoint();
    } else
      builder.create<mlir::cir::YieldOp>(addr->getLoc());
  }

  CurCGF = nullptr;
}
