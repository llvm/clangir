//===----- CIRGenItaniumCXXABI.cpp - Emit CIR from ASTs for a Module ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Itanium C++ ABI.  The class
// in this file generates structures that follow the Itanium C++ ABI, which is
// documented at:
//  https://itanium-cxx-abi.github.io/cxx-abi/abi.html
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
//
// It also supports the closely-related ARM ABI, documented at:
// https://developer.arm.com/documentation/ihi0041/g/
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCleanup.h"
#include "CIRGenFunctionInfo.h"
#include "ConstantInitBuilder.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/VTableBuilder.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;
using namespace clang;

namespace {
class CIRGenItaniumCXXABI : public cir::CIRGenCXXABI {
  /// All the vtables which have been defined.
  llvm::DenseMap<const CXXRecordDecl *, mlir::cir::GlobalOp> vTables;

protected:
  bool useArmMethodPtrAbi;
  bool useArmGuardVarAbi;
  bool use32BitVTableOffsetABI = false;

  ItaniumMangleContext &getMangleContext() {
    return cast<ItaniumMangleContext>(cir::CIRGenCXXABI::getMangleContext());
  }

  bool isVTableHidden(const CXXRecordDecl *rd) const {
    const auto &vtableLayout =
        CGM.getItaniumVTableContext().getVTableLayout(rd);

    for (const auto &vtableComponent : vtableLayout.vtable_components()) {
      if (vtableComponent.isRTTIKind()) {
        const CXXRecordDecl *rttiDecl = vtableComponent.getRTTIDecl();
        if (rttiDecl->getVisibility() == Visibility::HiddenVisibility)
          return true;
      } else if (vtableComponent.isUsedFunctionPointerKind()) {
        const CXXMethodDecl *method = vtableComponent.getFunctionDecl();
        if (method->getVisibility() == Visibility::HiddenVisibility &&
            !method->isDefined())
          return true;
      }
    }
    return false;
  }

  bool hasAnyUnusedVirtualInlineFunction(const CXXRecordDecl *rd) const {
    const auto &vtableLayout =
        CGM.getItaniumVTableContext().getVTableLayout(rd);

    for (const auto &vtableComponent : vtableLayout.vtable_components()) {
      // Skip empty slot.
      if (!vtableComponent.isUsedFunctionPointerKind())
        continue;

      const CXXMethodDecl *method = vtableComponent.getFunctionDecl();
      if (!method->getCanonicalDecl()->isInlined())
        continue;

      StringRef name = CGM.getMangledName(vtableComponent.getGlobalDecl());
      auto *op = CGM.getGlobalValue(name);
      if (auto globalOp = dyn_cast_or_null<mlir::cir::GlobalOp>(op))
        llvm_unreachable("NYI");

      if (auto funcOp = dyn_cast_or_null<mlir::cir::FuncOp>(op)) {
        // This checks if virtual inline function has already been emitted.
        // Note that it is possible that this inline function would be emitted
        // after trying to emit vtable speculatively. Because of this we do
        // an extra pass after emitting all deferred vtables to find and emit
        // these vtables opportunistically.
        if (!funcOp || funcOp.isDeclaration())
          return true;
      }
    }
    return false;
  }

public:
  CIRGenItaniumCXXABI(CIRGenModule &cgm, bool useArmMethodPtrAbi = false,
                      bool useArmGuardVarAbi = false)
      : CIRGenCXXABI(cgm), useArmMethodPtrAbi{useArmMethodPtrAbi},
        useArmGuardVarAbi{useArmGuardVarAbi} {
    assert(!useArmMethodPtrAbi && "NYI");
    assert(!useArmGuardVarAbi && "NYI");
  }
  AddedStructorArgs getImplicitConstructorArgs(CIRGenFunction &cgf,
                                               const CXXConstructorDecl *d,
                                               CXXCtorType type,
                                               bool forVirtualBase,
                                               bool delegating) override;

  bool NeedsVTTParameter(GlobalDecl gd) override;

  RecordArgABI getRecordArgABI(const clang::CXXRecordDecl *rd) const override {
    // If C++ prohibits us from making a copy, pass by address.
    if (!rd->canPassInRegisters())
      return RecordArgABI::Indirect;
    return RecordArgABI::Default;
  }

  bool classifyReturnType(CIRGenFunctionInfo &fi) const override;

  AddedStructorArgCounts
  buildStructorSignature(GlobalDecl gd,
                         llvm::SmallVectorImpl<CanQualType> &argTys) override;

  bool isThisCompleteObject(GlobalDecl gd) const override {
    // The Itanium ABI has separate complete-object vs. base-object variants of
    // both constructors and destructors.
    if (isa<CXXDestructorDecl>(gd.getDecl())) {
      llvm_unreachable("NYI");
    }
    if (isa<CXXConstructorDecl>(gd.getDecl())) {
      switch (gd.getCtorType()) {
      case Ctor_Complete:
        return true;

      case Ctor_Base:
        return false;

      case Ctor_CopyingClosure:
      case Ctor_DefaultClosure:
        llvm_unreachable("closure ctors in Itanium ABI?");

      case Ctor_Comdat:
        llvm_unreachable("emitting ctor comdat as function?");
      }
      llvm_unreachable("bad dtor kind");
    }

    // No other kinds.
    return false;
  }

  void buildInstanceFunctionProlog(CIRGenFunction &cgf) override;

  void addImplicitStructorParams(CIRGenFunction &cgf, QualType &resTy,
                                 FunctionArgList &params) override;

  mlir::Value getCXXDestructorImplicitParam(CIRGenFunction &cgf,
                                            const CXXDestructorDecl *dd,
                                            CXXDtorType type,
                                            bool forVirtualBase,
                                            bool delegating) override;
  void buildCXXConstructors(const clang::CXXConstructorDecl *d) override;
  void buildCXXDestructors(const clang::CXXDestructorDecl *d) override;
  void buildCXXStructor(clang::GlobalDecl gd) override;
  void buildDestructorCall(CIRGenFunction &cgf, const CXXDestructorDecl *dd,
                           CXXDtorType type, bool forVirtualBase,
                           bool delegating, Address theThis,
                           QualType thisTy) override;
  void registerGlobalDtor(CIRGenFunction &cgf, const VarDecl *d,
                          mlir::cir::FuncOp dtor,
                          mlir::Attribute addr) override;
  void buildRethrow(CIRGenFunction &cgf, bool isNoReturn) override;
  void buildThrow(CIRGenFunction &cgf, const CXXThrowExpr *e) override;
  CatchTypeInfo
  getAddrOfCXXCatchHandlerType(mlir::Location loc, QualType ty,
                               QualType catchHandlerType) override {
    auto rtti =
        dyn_cast<mlir::cir::GlobalViewAttr>(getAddrOfRTTIDescriptor(loc, ty));
    assert(rtti && "expected GlobalViewAttr");
    return CatchTypeInfo{rtti, 0};
  }

  void emitBeginCatch(CIRGenFunction &cgf, const CXXCatchStmt *c) override;

  bool canSpeculativelyEmitVTable(const CXXRecordDecl *rd) const override;
  mlir::cir::GlobalOp getAddrOfVTable(const CXXRecordDecl *rd,
                                      CharUnits vPtrOffset) override;
  CIRGenCallee getVirtualFunctionPointer(CIRGenFunction &cgf, GlobalDecl gd,
                                         Address theThis, mlir::Type ty,
                                         SourceLocation loc) override;
  mlir::Value getVTableAddressPoint(BaseSubobject base,
                                    const CXXRecordDecl *vTableClass) override;
  bool isVirtualOffsetNeededForVTableField(CIRGenFunction &cgf,
                                           CIRGenFunction::VPtr vptr) override;
  bool canSpeculativelyEmitVTableAsBaseClass(const CXXRecordDecl *rd) const;
  mlir::Value getVTableAddressPointInStructor(
      CIRGenFunction &cgf, const CXXRecordDecl *vTableClass, BaseSubobject base,
      const CXXRecordDecl *nearestVBase) override;
  void emitVTableDefinitions(CIRGenVTables &cgvt,
                             const CXXRecordDecl *rd) override;
  void emitVirtualInheritanceTables(const CXXRecordDecl *rd) override;
  mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc,
                                          QualType ty) override;
  bool useThunkForDtorVariant(const CXXDestructorDecl *dtor,
                              CXXDtorType dt) const override {
    // Itanium does not emit any destructor variant as an inline thunk.
    // Delegating may occur as an optimization, but all variants are either
    // emitted with external linkage or as linkonce if they are inline and used.
    return false;
  }

  StringRef getPureVirtualCallName() override { return "__cxa_pure_virtual"; }
  StringRef getDeletedVirtualCallName() override {
    return "__cxa_deleted_virtual";
  }

  /// TODO(cir): seems like could be shared between LLVM IR and CIR codegen.
  bool mayNeedDestruction(const VarDecl *vd) const {
    if (vd->needsDestruction(getContext()))
      return true;

    // If the variable has an incomplete class type (or array thereof), it
    // might need destruction.
    const Type *t = vd->getType()->getBaseElementTypeUnsafe();
    return t->getAs<RecordType>() && t->isIncompleteType();
  }

  /// Determine whether we will definitely emit this variable with a constant
  /// initializer, either because the language semantics demand it or because
  /// we know that the initializer is a constant.
  /// For weak definitions, any initializer available in the current translation
  /// is not necessarily reflective of the initializer used; such initializers
  /// are ignored unless if InspectInitForWeakDef is true.
  /// TODO(cir): seems like could be shared between LLVM IR and CIR codegen.
  bool
  isEmittedWithConstantInitializer(const VarDecl *vd,
                                   bool inspectInitForWeakDef = false) const {
    vd = vd->getMostRecentDecl();
    if (vd->hasAttr<ConstInitAttr>())
      return true;

    // All later checks examine the initializer specified on the variable. If
    // the variable is weak, such examination would not be correct.
    if (!inspectInitForWeakDef &&
        (vd->isWeak() || vd->hasAttr<SelectAnyAttr>()))
      return false;

    const VarDecl *initDecl = vd->getInitializingDeclaration();
    if (!initDecl)
      return false;

    // If there's no initializer to run, this is constant initialization.
    if (!initDecl->hasInit())
      return true;

    // If we have the only definition, we don't need a thread wrapper if we
    // will emit the value as a constant.
    if (isUniqueGVALinkage(getContext().GetGVALinkageForVariable(vd)))
      return !mayNeedDestruction(vd) && initDecl->evaluateValue();

    // Otherwise, we need a thread wrapper unless we know that every
    // translation unit will emit the value as a constant. We rely on the
    // variable being constant-initialized in every translation unit if it's
    // constant-initialized in any translation unit, which isn't actually
    // guaranteed by the standard but is necessary for sanity.
    return initDecl->hasConstantInitialization();
  }

  // TODO(cir): seems like could be shared between LLVM IR and CIR codegen.
  bool usesThreadWrapperFunction(const VarDecl *vd) const override {
    return !isEmittedWithConstantInitializer(vd) || mayNeedDestruction(vd);
  }

  bool doStructorsInitializeVPtrs(const CXXRecordDecl *vTableClass) override {
    return true;
  }

  size_t getSrcArgforCopyCtor(const CXXConstructorDecl *,
                              FunctionArgList &args) const override {
    assert(!args.empty() && "expected the arglist to not be empty!");
    return args.size() - 1;
  }

  void buildBadCastCall(CIRGenFunction &cgf, mlir::Location loc) override;

  // The traditional clang CodeGen emits calls to `__dynamic_cast` directly into
  // LLVM in the `emitDynamicCastCall` function. In CIR, `dynamic_cast`
  // expressions are lowered to `cir.dyn_cast` ops instead of calls to runtime
  // functions. So during CIRGen we don't need the `emitDynamicCastCall`
  // function that clang CodeGen has.

  mlir::Value buildDynamicCast(CIRGenFunction &cgf, mlir::Location loc,
                               QualType srcRecordTy, QualType destRecordTy,
                               mlir::cir::PointerType destCirTy, bool isRefCast,
                               Address src) override;

  mlir::cir::MethodAttr
  buildVirtualMethodAttr(mlir::cir::MethodType methodTy,
                         const CXXMethodDecl *md) override;

  /**************************** RTTI Uniqueness ******************************/
protected:
  /// Returns true if the ABI requires RTTI type_info objects to be unique
  /// across a program.
  virtual bool shouldRTTIBeUnique() const { return true; }

public:
  /// What sort of unique-RTTI behavior should we use?
  enum RTTIUniquenessKind {
    /// We are guaranteeing, or need to guarantee, that the RTTI string
    /// is unique.
    RUK_Unique,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// can demote to hidden visibility but must use string comparisons.
    RUK_NonUniqueHidden,

    /// We are not guaranteeing uniqueness for the RTTI string, so we
    /// have to use string comparisons, but we also have to emit it with
    /// non-hidden visibility.
    RUK_NonUniqueVisible
  };

  /// Return the required visibility status for the given type and linkage in
  /// the current ABI.
  RTTIUniquenessKind
  classifyRTTIUniqueness(QualType canTy,
                         mlir::cir::GlobalLinkageKind linkage) const;
  friend class CIRGenItaniumRTTIBuilder;
};
} // namespace

CIRGenCXXABI::AddedStructorArgs CIRGenItaniumCXXABI::getImplicitConstructorArgs(
    CIRGenFunction &cgf, const CXXConstructorDecl *d, CXXCtorType type,
    bool forVirtualBase, bool delegating) {
  assert(!NeedsVTTParameter(GlobalDecl(d, type)) && "VTT NYI");

  return {};
}

/// Return whether the given global decl needs a VTT parameter, which it does if
/// it's a base constructor or destructor with virtual bases.
bool CIRGenItaniumCXXABI::NeedsVTTParameter(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  // We don't have any virtual bases, just return early.
  if (!md->getParent()->getNumVBases())
    return false;

  // Check if we have a base constructor.
  if (isa<CXXConstructorDecl>(md) && gd.getCtorType() == Ctor_Base)
    return true;

  // Check if we have a base destructor.
  if (isa<CXXDestructorDecl>(md) && gd.getDtorType() == Dtor_Base)
    llvm_unreachable("NYI");

  return false;
}

CIRGenCXXABI *cir::CreateCIRGenItaniumCXXABI(CIRGenModule &cgm) {
  switch (cgm.getASTContext().getCXXABIKind()) {
  case TargetCXXABI::GenericItanium:
  case TargetCXXABI::GenericAArch64:
  case TargetCXXABI::AppleARM64:
    // TODO: this isn't quite right, clang uses AppleARM64CXXABI which inherits
    // from ARMCXXABI. We'll have to follow suit.
    assert(!MissingFeatures::appleArm64CXXABI());
    return new CIRGenItaniumCXXABI(cgm);

  default:
    llvm_unreachable("bad or NYI ABI kind");
  }
}

bool CIRGenItaniumCXXABI::classifyReturnType(CIRGenFunctionInfo &fi) const {
  auto *rd = fi.getReturnType()->getAsCXXRecordDecl();
  assert(!rd && "RecordDecl return types NYI");
  return false;
}

CIRGenCXXABI::AddedStructorArgCounts
CIRGenItaniumCXXABI::buildStructorSignature(
    GlobalDecl gd, llvm::SmallVectorImpl<CanQualType> &argTys) {
  auto &context = getContext();

  // All parameters are already in place except VTT, which goes after 'this'.
  // These are clang types, so we don't need to worry about sret yet.

  // Check if we need to add a VTT parameter (which has type void **).
  if ((isa<CXXConstructorDecl>(gd.getDecl()) ? gd.getCtorType() == Ctor_Base
                                             : gd.getDtorType() == Dtor_Base) &&
      cast<CXXMethodDecl>(gd.getDecl())->getParent()->getNumVBases() != 0) {
    llvm_unreachable("NYI");
    (void)context;
  }

  return AddedStructorArgCounts{};
}

// Find out how to cirgen the complete destructor and constructor
namespace {
enum class StructorCIRGen { Emit, RAUW, Alias, COMDAT };
} // namespace

static StructorCIRGen getCIRGenToUse(CIRGenModule &cgm,
                                     const CXXMethodDecl *md) {
  if (!cgm.getCodeGenOpts().CXXCtorDtorAliases)
    return StructorCIRGen::Emit;

  // The complete and base structors are not equivalent if there are any virtual
  // bases, so emit separate functions.
  if (md->getParent()->getNumVBases())
    return StructorCIRGen::Emit;

  GlobalDecl aliasDecl;
  if (const auto *dd = dyn_cast<CXXDestructorDecl>(md)) {
    aliasDecl = GlobalDecl(dd, Dtor_Complete);
  } else {
    const auto *cd = cast<CXXConstructorDecl>(md);
    aliasDecl = GlobalDecl(cd, Ctor_Complete);
  }
  auto linkage = cgm.getFunctionLinkage(aliasDecl);
  (void)linkage;

  if (mlir::cir::isDiscardableIfUnused(linkage))
    return StructorCIRGen::RAUW;

  // FIXME: Should we allow available_externally aliases?
  if (!mlir::cir::isValidLinkage(linkage))
    return StructorCIRGen::RAUW;

  if (mlir::cir::isWeakForLinker(linkage)) {
    // Only ELF and wasm support COMDATs with arbitrary names (C5/D5).
    if (cgm.getTarget().getTriple().isOSBinFormatELF() ||
        cgm.getTarget().getTriple().isOSBinFormatWasm())
      return StructorCIRGen::COMDAT;
    return StructorCIRGen::Emit;
  }

  return StructorCIRGen::Alias;
}

static void emitConstructorDestructorAlias(CIRGenModule &cgm,
                                           GlobalDecl aliasDecl,
                                           GlobalDecl targetDecl) {
  auto linkage = cgm.getFunctionLinkage(aliasDecl);

  // Does this function alias already exists?
  StringRef mangledName = cgm.getMangledName(aliasDecl);
  auto globalValue = dyn_cast_or_null<mlir::cir::CIRGlobalValueInterface>(
      cgm.getGlobalValue(mangledName));
  if (globalValue && !globalValue.isDeclaration()) {
    return;
  }

  auto entry =
      dyn_cast_or_null<mlir::cir::FuncOp>(cgm.getGlobalValue(mangledName));

  // Retrieve aliasee info.
  auto aliasee =
      dyn_cast_or_null<mlir::cir::FuncOp>(cgm.GetAddrOfGlobal(targetDecl));
  assert(aliasee && "expected cir.func");

  // Populate actual alias.
  cgm.buildAliasForGlobal(mangledName, entry, aliasDecl, aliasee, linkage);
}

void CIRGenItaniumCXXABI::buildCXXStructor(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());
  auto *cd = dyn_cast<CXXConstructorDecl>(md);
  const CXXDestructorDecl *dd = cd ? nullptr : cast<CXXDestructorDecl>(md);

  StructorCIRGen cirGenType = getCIRGenToUse(CGM, md);

  if (cd ? gd.getCtorType() == Ctor_Complete
         : gd.getDtorType() == Dtor_Complete) {
    GlobalDecl baseDecl;
    if (cd)
      baseDecl = gd.getWithCtorType(Ctor_Base);
    else
      baseDecl = gd.getWithDtorType(Dtor_Base);

    if (cirGenType == StructorCIRGen::Alias ||
        cirGenType == StructorCIRGen::COMDAT) {
      emitConstructorDestructorAlias(CGM, gd, baseDecl);
      return;
    }

    if (cirGenType == StructorCIRGen::RAUW) {
      StringRef mangledName = CGM.getMangledName(gd);
      auto *aliasee = CGM.GetAddrOfGlobal(baseDecl);
      CGM.addReplacement(mangledName, aliasee);
      return;
    }
  }

  // The base destructor is equivalent to the base destructor of its base class
  // if there is exactly one non-virtual base class with a non-trivial
  // destructor, there are no fields with a non-trivial destructor, and the body
  // of the destructor is trivial.
  if (dd && gd.getDtorType() == Dtor_Base &&
      cirGenType != StructorCIRGen::COMDAT &&
      !CGM.tryEmitBaseDestructorAsAlias(dd))
    return;

  // FIXME: The deleting destructor is equivalent to the selected operator
  // delete if:
  //  * either the delete is a destroying operator delete or the destructor
  //    would be trivial if it weren't virtual.
  //  * the conversion from the 'this' parameter to the first parameter of the
  //    destructor is equivalent to a bitcast,
  //  * the destructor does not have an implicit "this" return, and
  //  * the operator delete has the same calling convention and CIR function
  //    type as the destructor.
  // In such cases we should try to emit the deleting dtor as an alias to the
  // selected 'operator delete'.

  auto fn = CGM.codegenCXXStructor(gd);

  if (cirGenType == StructorCIRGen::COMDAT) {
    llvm_unreachable("NYI");
  } else {
    CGM.maybeSetTrivialComdat(*md, fn);
  }
}

void CIRGenItaniumCXXABI::addImplicitStructorParams(CIRGenFunction &cgf,
                                                    QualType &resTy,
                                                    FunctionArgList &params) {
  const auto *md = cast<CXXMethodDecl>(cgf.CurGD.getDecl());
  assert(isa<CXXConstructorDecl>(md) || isa<CXXDestructorDecl>(md));

  // Check if we need a VTT parameter as well.
  if (NeedsVTTParameter(cgf.CurGD)) {
    llvm_unreachable("NYI");
  }
}

mlir::Value CIRGenCXXABI::loadIncomingCXXThis(CIRGenFunction &cgf) {
  return cgf.createLoad(getThisDecl(cgf), "this");
}

void CIRGenCXXABI::setCXXABIThisValue(CIRGenFunction &cgf,
                                      mlir::Value thisPtr) {
  /// Initialize the 'this' slot.
  assert(getThisDecl(cgf) && "no 'this' variable for function");
  cgf.CXXABIThisValue = thisPtr;
}

void CIRGenItaniumCXXABI::buildInstanceFunctionProlog(CIRGenFunction &cgf) {
  // Naked functions have no prolog.
  if (cgf.CurFuncDecl && cgf.CurFuncDecl->hasAttr<NakedAttr>())
    llvm_unreachable("NYI");

  /// Initialize the 'this' slot. In the Itanium C++ ABI, no prologue
  /// adjustments are required, because they are all handled by thunks.
  setCXXABIThisValue(cgf, loadIncomingCXXThis(cgf));

  /// Initialize the 'vtt' slot if needed.
  if (getStructorImplicitParamDecl(cgf)) {
    llvm_unreachable("NYI");
  }

  /// If this is a function that the ABI specifies returns 'this', initialize
  /// the return slot to this' at the start of the function.
  ///
  /// Unlike the setting of return types, this is done within the ABI
  /// implementation instead of by clients of CIRGenCXXBI because:
  /// 1) getThisValue is currently protected
  /// 2) in theory, an ABI could implement 'this' returns some other way;
  ///    HasThisReturn only specifies a contract, not the implementation
  if (HasThisReturn(cgf.CurGD))
    llvm_unreachable("NYI");
}

void CIRGenItaniumCXXABI::buildCXXConstructors(const CXXConstructorDecl *d) {
  // Just make sure we're in sync with TargetCXXABI.
  assert(CGM.getTarget().getCXXABI().hasConstructorVariants());

  // The constructor used for constructing this as a base class;
  // ignores virtual bases.
  CGM.buildGlobal(GlobalDecl(d, Ctor_Base));

  // The constructor used for constructing this as a complete class;
  // constructs the virtual bases, then calls the base constructor.
  if (!d->getParent()->isAbstract()) {
    // We don't need to emit the complete ctro if the class is abstract.
    CGM.buildGlobal(GlobalDecl(d, Ctor_Complete));
  }
}

void CIRGenItaniumCXXABI::buildCXXDestructors(const CXXDestructorDecl *d) {
  // The destructor used for destructing this as a base class; ignores
  // virtual bases.
  CGM.buildGlobal(GlobalDecl(d, Dtor_Base));

  // The destructor used for destructing this as a most-derived class;
  // call the base destructor and then destructs any virtual bases.
  CGM.buildGlobal(GlobalDecl(d, Dtor_Complete));

  // The destructor in a virtual table is always a 'deleting'
  // destructor, which calls the complete destructor and then uses the
  // appropriate operator delete.
  if (d->isVirtual())
    CGM.buildGlobal(GlobalDecl(d, Dtor_Deleting));
}

namespace {
/// From traditional LLVM, useful info for LLVM lowering support:
/// A cleanup to call __cxa_end_catch.  In many cases, the caught
/// exception type lets us state definitively that the thrown exception
/// type does not have a destructor.  In particular:
///   - Catch-alls tell us nothing, so we have to conservatively
///     assume that the thrown exception might have a destructor.
///   - Catches by reference behave according to their base types.
///   - Catches of non-record types will only trigger for exceptions
///     of non-record types, which never have destructors.
///   - Catches of record types can trigger for arbitrary subclasses
///     of the caught type, so we have to assume the actual thrown
///     exception type might have a throwing destructor, even if the
///     caught type's destructor is trivial or nothrow.
struct CallEndCatch final : EHScopeStack::Cleanup {
  CallEndCatch(bool mightThrow) : mightThrow(mightThrow) {}
  bool mightThrow;

  void Emit(CIRGenFunction &cgf, Flags flags) override {
    if (!mightThrow) {
      // Traditional LLVM codegen would emit a call to __cxa_end_catch
      // here. For CIR, just let it pass since the cleanup is going
      // to be emitted on a later pass when lowering the catch region.
      // CGF.EmitNounwindRuntimeCall(getEndCatchFn(CGF.CGM));
      cgf.getBuilder().create<mlir::cir::YieldOp>(*cgf.currSrcLoc);
      return;
    }

    // Traditional LLVM codegen would emit a call to __cxa_end_catch
    // here. For CIR, just let it pass since the cleanup is going
    // to be emitted on a later pass when lowering the catch region.
    // CGF.EmitRuntimeCallOrTryCall(getEndCatchFn(CGF.CGM));
    cgf.getBuilder().create<mlir::cir::YieldOp>(*cgf.currSrcLoc);
  }
};
} // namespace

/// From traditional LLVM codegen, useful info for LLVM lowering support:
/// Emits a call to __cxa_begin_catch and enters a cleanup to call
/// __cxa_end_catch. If -fassume-nothrow-exception-dtor is specified, we assume
/// that the exception object's dtor is nothrow, therefore the __cxa_end_catch
/// call can be marked as nounwind even if EndMightThrow is true.
///
/// \param EndMightThrow - true if __cxa_end_catch might throw
static mlir::Value callBeginCatch(CIRGenFunction &cgf, mlir::Type paramTy,
                                  bool endMightThrow) {
  auto catchParam = cgf.getBuilder().create<mlir::cir::CatchParamOp>(
      cgf.getBuilder().getUnknownLoc(), paramTy, nullptr, nullptr);

  cgf.EHStack.pushCleanup<CallEndCatch>(
      NormalAndEHCleanup,
      endMightThrow && !cgf.cgm.getLangOpts().AssumeNothrowExceptionDtor);

  return catchParam.getParam();
}

/// A "special initializer" callback for initializing a catch
/// parameter during catch initialization.
static void initCatchParam(CIRGenFunction &cgf, const VarDecl &catchParam,
                           Address paramAddr, SourceLocation loc) {
  CanQualType catchType =
      cgf.cgm.getASTContext().getCanonicalType(catchParam.getType());
  auto cirCatchTy = cgf.convertTypeForMem(catchType);

  // If we're catching by reference, we can just cast the object
  // pointer to the appropriate pointer.
  if (isa<ReferenceType>(catchType)) {
    llvm_unreachable("NYI");
    return;
  }

  // Scalars and complexes.
  TypeEvaluationKind tek = cgf.getEvaluationKind(catchType);
  if (tek != TEK_Aggregate) {
    // Notes for LLVM lowering:
    // If the catch type is a pointer type, __cxa_begin_catch returns
    // the pointer by value.
    if (catchType->hasPointerRepresentation()) {
      auto catchParam = callBeginCatch(cgf, cirCatchTy, false);

      switch (catchType.getQualifiers().getObjCLifetime()) {
      case Qualifiers::OCL_Strong:
        llvm_unreachable("NYI");
        // arc retain non block:
        assert(!MissingFeatures::arc());
        [[fallthrough]];

      case Qualifiers::OCL_None:
      case Qualifiers::OCL_ExplicitNone:
      case Qualifiers::OCL_Autoreleasing:
        cgf.getBuilder().createStore(cgf.getBuilder().getUnknownLoc(),
                                     catchParam, paramAddr);
        return;

      case Qualifiers::OCL_Weak:
        llvm_unreachable("NYI");
        // arc init weak:
        assert(!MissingFeatures::arc());
        return;
      }
      llvm_unreachable("bad ownership qualifier!");
    }

    // Otherwise, it returns a pointer into the exception object.
    auto catchParam =
        callBeginCatch(cgf, cgf.getBuilder().getPointerTo(cirCatchTy), false);
    LValue srcLV = cgf.MakeNaturalAlignAddrLValue(catchParam, catchType);
    LValue destLV = cgf.makeAddrLValue(paramAddr, catchType);
    switch (tek) {
    case TEK_Complex:
      llvm_unreachable("NYI");
      return;
    case TEK_Scalar: {
      auto exnLoad = cgf.buildLoadOfScalar(srcLV, catchParam.getLoc());
      cgf.buildStoreOfScalar(exnLoad, destLV, /*init*/ true);
      return;
    }
    case TEK_Aggregate:
      llvm_unreachable("evaluation kind filtered out!");
    }
    llvm_unreachable("bad evaluation kind");
  }

  // Check for a copy expression.  If we don't have a copy expression,
  // that means a trivial copy is okay.
  const Expr *copyExpr = catchParam.getInit();
  if (!copyExpr) {
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
}

/// Begins a catch statement by initializing the catch variable and
/// calling __cxa_begin_catch.
void CIRGenItaniumCXXABI::emitBeginCatch(CIRGenFunction &cgf,
                                         const CXXCatchStmt *c) {
  // Notes for LLVM lowering:
  // We have to be very careful with the ordering of cleanups here:
  //   C++ [except.throw]p4:
  //     The destruction [of the exception temporary] occurs
  //     immediately after the destruction of the object declared in
  //     the exception-declaration in the handler.
  //
  // So the precise ordering is:
  //   1.  Construct catch variable.
  //   2.  __cxa_begin_catch
  //   3.  Enter __cxa_end_catch cleanup
  //   4.  Enter dtor cleanup
  //
  // We do this by using a slightly abnormal initialization process.
  // Delegation sequence:
  //   - ExitCXXTryStmt opens a RunCleanupsScope
  //     - EmitAutoVarAlloca creates the variable and debug info
  //       - InitCatchParam initializes the variable from the exception
  //       - CallBeginCatch calls __cxa_begin_catch
  //       - CallBeginCatch enters the __cxa_end_catch cleanup
  //     - EmitAutoVarCleanups enters the variable destructor cleanup
  //   - EmitCXXTryStmt emits the code for the catch body
  //   - EmitCXXTryStmt close the RunCleanupsScope

  VarDecl *catchParam = c->getExceptionDecl();
  if (!catchParam) {
    callBeginCatch(cgf, cgf.getBuilder().getVoidPtrTy(), true);
    return;
  }

  auto getCatchParamAllocaIP = [&]() {
    auto currIns = cgf.getBuilder().saveInsertionPoint();
    auto *currParent = currIns.getBlock()->getParentOp();
    mlir::Operation *scopeLikeOp =
        currParent->getParentOfType<mlir::cir::ScopeOp>();
    if (!scopeLikeOp)
      scopeLikeOp = currParent->getParentOfType<mlir::cir::FuncOp>();
    assert(scopeLikeOp && "unknown outermost scope-like parent");
    assert(scopeLikeOp->getNumRegions() == 1 && "expected single region");

    auto *insertBlock = &scopeLikeOp->getRegion(0).getBlocks().back();
    return cgf.getBuilder().getBestAllocaInsertPoint(insertBlock);
  };

  // Emit the local. Make sure the alloca's superseed the current scope, since
  // these are going to be consumed by `cir.catch`, which is not within the
  // current scope.
  auto var = cgf.buildAutoVarAlloca(*catchParam, getCatchParamAllocaIP());
  initCatchParam(cgf, *catchParam, var.getObjectAddress(cgf), c->getBeginLoc());
  // FIXME(cir): double check cleanups here are happening in the right blocks.
  cgf.buildAutoVarCleanups(var);
}

mlir::cir::GlobalOp
CIRGenItaniumCXXABI::getAddrOfVTable(const CXXRecordDecl *rd,
                                     CharUnits vPtrOffset) {
  assert(vPtrOffset.isZero() && "Itanium ABI only supports zero vptr offsets");
  mlir::cir::GlobalOp &vtable = vTables[rd];
  if (vtable)
    return vtable;

  // Queue up this vtable for possible deferred emission.
  CGM.addDeferredVTable(rd);

  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  getMangleContext().mangleCXXVTable(rd, out);

  const VTableLayout &vtLayout =
      CGM.getItaniumVTableContext().getVTableLayout(rd);
  auto vTableType = CGM.getVTables().getVTableType(vtLayout);

  // Use pointer alignment for the vtable. Otherwise we would align them based
  // on the size of the initializer which doesn't make sense as only single
  // values are read.
  unsigned pAlign = CGM.getItaniumVTableContext().isRelativeLayout()
                        ? 32
                        : CGM.getTarget().getPointerAlign(LangAS::Default);

  vtable = CGM.createOrReplaceCXXRuntimeVariable(
      CGM.getLoc(rd->getSourceRange()), name, vTableType,
      mlir::cir::GlobalLinkageKind::ExternalLinkage,
      getContext().toCharUnitsFromBits(pAlign));
  // LLVM codegen handles unnamedAddr
  assert(!MissingFeatures::unnamedAddr());

  // In MS C++ if you have a class with virtual functions in which you are using
  // selective member import/export, then all virtual functions must be exported
  // unless they are inline, otherwise a link error will result. To match this
  // behavior, for such classes, we dllimport the vtable if it is defined
  // externally and all the non-inline virtual methods are marked dllimport, and
  // we dllexport the vtable if it is defined in this TU and all the non-inline
  // virtual methods are marked dllexport.
  if (CGM.getTarget().hasPS4DLLImportExport())
    llvm_unreachable("NYI");

  CGM.setGVProperties(vtable, rd);
  return vtable;
}

CIRGenCallee CIRGenItaniumCXXABI::getVirtualFunctionPointer(
    CIRGenFunction &cgf, GlobalDecl gd, Address theThis, mlir::Type ty,
    SourceLocation loc) {
  auto mlirLoc = cgf.getLoc(loc);
  auto tyPtr = cgf.getBuilder().getPointerTo(ty);
  auto *methodDecl = cast<CXXMethodDecl>(gd.getDecl());
  auto vTable =
      cgf.getVTablePtr(mlirLoc, theThis, cgf.getBuilder().getPointerTo(tyPtr),
                       methodDecl->getParent());

  uint64_t vTableIndex = CGM.getItaniumVTableContext().getMethodVTableIndex(gd);
  mlir::Value vFunc{};
  if (cgf.shouldEmitVTableTypeCheckedLoad(methodDecl->getParent())) {
    llvm_unreachable("NYI");
  } else {
    cgf.buildTypeMetadataCodeForVCall(methodDecl->getParent(), vTable, loc);

    mlir::Value vFuncLoad;
    if (CGM.getItaniumVTableContext().isRelativeLayout()) {
      llvm_unreachable("NYI");
    } else {
      vTable = cgf.getBuilder().createBitcast(
          mlirLoc, vTable, cgf.getBuilder().getPointerTo(tyPtr));
      auto vTableSlotPtr =
          cgf.getBuilder().create<mlir::cir::VTableAddrPointOp>(
              mlirLoc, cgf.getBuilder().getPointerTo(tyPtr),
              ::mlir::FlatSymbolRefAttr{}, vTable,
              /*vtable_index=*/0, vTableIndex);
      vFuncLoad = cgf.getBuilder().createAlignedLoad(
          mlirLoc, tyPtr, vTableSlotPtr, cgf.getPointerAlign());
    }

    // Add !invariant.load md to virtual function load to indicate that
    // function didn't change inside vtable.
    // It's safe to add it without -fstrict-vtable-pointers, but it would not
    // help in devirtualization because it will only matter if we will have 2
    // the same virtual function loads from the same vtable load, which won't
    // happen without enabled devirtualization with -fstrict-vtable-pointers.
    if (CGM.getCodeGenOpts().OptimizationLevel > 0 &&
        CGM.getCodeGenOpts().StrictVTablePointers) {
      llvm_unreachable("NYI");
    }
    vFunc = vFuncLoad;
  }

  CIRGenCallee callee(gd, vFunc.getDefiningOp());
  return callee;
}

mlir::Value
CIRGenItaniumCXXABI::getVTableAddressPoint(BaseSubobject base,
                                           const CXXRecordDecl *vTableClass) {
  auto vtable = getAddrOfVTable(vTableClass, CharUnits());

  // Find the appropriate vtable within the vtable group, and the address point
  // within that vtable.
  VTableLayout::AddressPointLocation addressPoint =
      CGM.getItaniumVTableContext()
          .getVTableLayout(vTableClass)
          .getAddressPoint(base);

  auto &builder = CGM.getBuilder();
  auto vtablePtrTy = builder.getVirtualFnPtrType(/*isVarArg=*/false);

  return builder.create<mlir::cir::VTableAddrPointOp>(
      CGM.getLoc(vTableClass->getSourceRange()), vtablePtrTy,
      mlir::FlatSymbolRefAttr::get(vtable.getSymNameAttr()), mlir::Value{},
      addressPoint.VTableIndex, addressPoint.AddressPointIndex);
}

mlir::Value CIRGenItaniumCXXABI::getVTableAddressPointInStructor(
    CIRGenFunction &cgf, const CXXRecordDecl *vTableClass, BaseSubobject base,
    const CXXRecordDecl *nearestVBase) {

  if ((base.getBase()->getNumVBases() || nearestVBase != nullptr) &&
      NeedsVTTParameter(cgf.CurGD)) {
    llvm_unreachable("NYI");
  }
  return getVTableAddressPoint(base, vTableClass);
}

bool CIRGenItaniumCXXABI::isVirtualOffsetNeededForVTableField(
    CIRGenFunction &cgf, CIRGenFunction::VPtr vptr) {
  if (vptr.NearestVBase == nullptr)
    return false;
  return NeedsVTTParameter(cgf.CurGD);
}

bool CIRGenItaniumCXXABI::canSpeculativelyEmitVTableAsBaseClass(
    const CXXRecordDecl *rd) const {
  // We don't emit available_externally vtables if we are in -fapple-kext mode
  // because kext mode does not permit devirtualization.
  if (CGM.getLangOpts().AppleKext)
    return false;

  // If the vtable is hidden then it is not safe to emit an available_externally
  // copy of vtable.
  if (isVTableHidden(rd))
    return false;

  if (CGM.getCodeGenOpts().ForceEmitVTables)
    return true;

  // If we don't have any not emitted inline virtual function then we are safe
  // to emit an available_externally copy of vtable.
  // FIXME we can still emit a copy of the vtable if we
  // can emit definition of the inline functions.
  if (hasAnyUnusedVirtualInlineFunction(rd))
    return false;

  // For a class with virtual bases, we must also be able to speculatively
  // emit the VTT, because CodeGen doesn't have separate notions of "can emit
  // the vtable" and "can emit the VTT". For a base subobject, this means we
  // need to be able to emit non-virtual base vtables.
  if (rd->getNumVBases()) {
    for (const auto &b : rd->bases()) {
      auto *brd = b.getType()->getAsCXXRecordDecl();
      assert(brd && "no class for base specifier");
      if (b.isVirtual() || !brd->isDynamicClass())
        continue;
      if (!canSpeculativelyEmitVTableAsBaseClass(brd))
        return false;
    }
  }

  return true;
}

bool CIRGenItaniumCXXABI::canSpeculativelyEmitVTable(
    const CXXRecordDecl *rd) const {
  if (!canSpeculativelyEmitVTableAsBaseClass(rd))
    return false;

  // For a complete-object vtable (or more specifically, for the VTT), we need
  // to be able to speculatively emit the vtables of all dynamic virtual bases.
  for (const auto &b : rd->vbases()) {
    auto *brd = b.getType()->getAsCXXRecordDecl();
    assert(brd && "no class for base specifier");
    if (!brd->isDynamicClass())
      continue;
    if (!canSpeculativelyEmitVTableAsBaseClass(brd))
      return false;
  }

  return true;
}

namespace {
class CIRGenItaniumRTTIBuilder {
  CIRGenModule &cgm;                 // Per-module state.
  const CIRGenItaniumCXXABI &cxxabi; // Per-module state.

  /// The fields of the RTTI descriptor currently being built.
  SmallVector<mlir::Attribute, 16> fields;

  // Returns the mangled type name of the given type.
  mlir::cir::GlobalOp getAddrOfTypeName(mlir::Location loc, QualType ty,
                                        mlir::cir::GlobalLinkageKind linkage);

  // /// Returns the constant for the RTTI
  // /// descriptor of the given type.
  mlir::Attribute getAddrOfExternalRttiDescriptor(mlir::Location loc,
                                                  QualType ty);

  /// Build the vtable pointer for the given type.
  void buildVTablePointer(mlir::Location loc, const Type *ty);

  /// Build an abi::__si_class_type_info, used for single inheritance, according
  /// to the Itanium C++ ABI, 2.9.5p6b.
  void buildSiClassTypeInfo(mlir::Location loc, const CXXRecordDecl *rd);

  /// Build an abi::__vmi_class_type_info, used for
  /// classes with bases that do not satisfy the abi::__si_class_type_info
  /// constraints, according ti the Itanium C++ ABI, 2.9.5p5c.
  void buildVmiClassTypeInfo(mlir::Location loc, const CXXRecordDecl *rd);

  // /// Build an abi::__pointer_type_info struct, used
  // /// for pointer types.
  // void BuildPointerTypeInfo(QualType PointeeTy);

  // /// Build the appropriate kind of
  // /// type_info for an object type.
  // void BuildObjCObjectTypeInfo(const ObjCObjectType *Ty);

  // /// Build an
  // abi::__pointer_to_member_type_info
  // /// struct, used for member pointer types.
  // void BuildPointerToMemberTypeInfo(const MemberPointerType *Ty);

public:
  CIRGenItaniumRTTIBuilder(const CIRGenItaniumCXXABI &abi, CIRGenModule &cgm)
      : cgm(cgm), cxxabi(abi) {}

  // Pointer type info flags.
  enum {
    /// PTI_Const - Type has const qualifier.
    PTI_Const = 0x1,

    /// PTI_Volatile - Type has volatile qualifier.
    PTI_Volatile = 0x2,

    /// PTI_Restrict - Type has restrict qualifier.
    PTI_Restrict = 0x4,

    /// PTI_Incomplete - Type is incomplete.
    PTI_Incomplete = 0x8,

    /// PTI_ContainingClassIncomplete - Containing class is incomplete.
    /// (in pointer to member).
    PTI_ContainingClassIncomplete = 0x10,

    /// PTI_TransactionSafe - Pointee is transaction_safe function (C++ TM TS).
    // PTI_TransactionSafe = 0x20,

    /// PTI_Noexcept - Pointee is noexcept function (C++1z).
    PTI_Noexcept = 0x40,
  };

  // VMI type info flags.
  enum {
    /// VMI_NonDiamondRepeat - Class has non-diamond repeated inheritance.
    VMI_NonDiamondRepeat = 0x1,

    /// VMI_DiamondShaped - Class is diamond shaped.
    VMI_DiamondShaped = 0x2
  };

  // Base class type info flags.
  enum {
    /// BCTI_Virtual - Base class is virtual.
    BCTI_Virtual = 0x1,

    /// BCTI_Public - Base class is public.
    BCTI_Public = 0x2
  };

  /// Build the RTTI type info struct for the given type, or
  /// link to an existing RTTI descriptor if one already exists.
  mlir::Attribute buildTypeInfo(mlir::Location loc, QualType ty);

  /// Build the RTTI type info struct for the given type.
  mlir::Attribute buildTypeInfo(mlir::Location loc, QualType ty,
                                mlir::cir::GlobalLinkageKind linkage,
                                mlir::SymbolTable::Visibility visibility);
};
} // namespace

/// Given a builtin type, returns whether the type
/// info for that type is defined in the standard library.
/// TODO(cir): this can unified with LLVM codegen
static bool typeInfoIsInStandardLibrary(const BuiltinType *ty) {
  // Itanium C++ ABI 2.9.2:
  //   Basic type information (e.g. for "int", "bool", etc.) will be kept in
  //   the run-time support library. Specifically, the run-time support
  //   library should contain type_info objects for the types X, X* and
  //   X const*, for every X in: void, std::nullptr_t, bool, wchar_t, char,
  //   unsigned char, signed char, short, unsigned short, int, unsigned int,
  //   long, unsigned long, long long, unsigned long long, float, double,
  //   long double, char16_t, char32_t, and the IEEE 754r decimal and
  //   half-precision floating point types.
  //
  // GCC also emits RTTI for __int128.
  // FIXME: We do not emit RTTI information for decimal types here.

  // Types added here must also be added to EmitFundamentalRTTIDescriptors.
  switch (ty->getKind()) {
  case BuiltinType::WasmExternRef:
  case BuiltinType::HLSLResource:
    llvm_unreachable("NYI");
  case BuiltinType::Void:
  case BuiltinType::NullPtr:
  case BuiltinType::Bool:
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
  case BuiltinType::Char_U:
  case BuiltinType::Char_S:
  case BuiltinType::UChar:
  case BuiltinType::SChar:
  case BuiltinType::Short:
  case BuiltinType::UShort:
  case BuiltinType::Int:
  case BuiltinType::UInt:
  case BuiltinType::Long:
  case BuiltinType::ULong:
  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
  case BuiltinType::Half:
  case BuiltinType::Float:
  case BuiltinType::Double:
  case BuiltinType::LongDouble:
  case BuiltinType::Float16:
  case BuiltinType::Float128:
  case BuiltinType::Ibm128:
  case BuiltinType::Char8:
  case BuiltinType::Char16:
  case BuiltinType::Char32:
  case BuiltinType::Int128:
  case BuiltinType::UInt128:
    return true;

#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:
#include "clang/Basic/OpenCLImageTypes.def"
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext) case BuiltinType::Id:
#include "clang/Basic/OpenCLExtensionTypes.def"
  case BuiltinType::OCLSampler:
  case BuiltinType::OCLEvent:
  case BuiltinType::OCLClkEvent:
  case BuiltinType::OCLQueue:
  case BuiltinType::OCLReserveID:
#define SVE_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/AArch64SVEACLETypes.def"
#define PPC_VECTOR_TYPE(Name, Id, Size) case BuiltinType::Id:
#include "clang/Basic/PPCTypes.def"
#define RVV_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/RISCVVTypes.def"
#define AMDGPU_TYPE(Name, Id, SingletonId) case BuiltinType::Id:
#include "clang/Basic/AMDGPUTypes.def"
  case BuiltinType::ShortAccum:
  case BuiltinType::Accum:
  case BuiltinType::LongAccum:
  case BuiltinType::UShortAccum:
  case BuiltinType::UAccum:
  case BuiltinType::ULongAccum:
  case BuiltinType::ShortFract:
  case BuiltinType::Fract:
  case BuiltinType::LongFract:
  case BuiltinType::UShortFract:
  case BuiltinType::UFract:
  case BuiltinType::ULongFract:
  case BuiltinType::SatShortAccum:
  case BuiltinType::SatAccum:
  case BuiltinType::SatLongAccum:
  case BuiltinType::SatUShortAccum:
  case BuiltinType::SatUAccum:
  case BuiltinType::SatULongAccum:
  case BuiltinType::SatShortFract:
  case BuiltinType::SatFract:
  case BuiltinType::SatLongFract:
  case BuiltinType::SatUShortFract:
  case BuiltinType::SatUFract:
  case BuiltinType::SatULongFract:
  case BuiltinType::BFloat16:
    return false;

  case BuiltinType::Dependent:
#define BUILTIN_TYPE(Id, SingletonId)
#define PLACEHOLDER_TYPE(Id, SingletonId) case BuiltinType::Id:
#include "clang/AST/BuiltinTypes.def"
    llvm_unreachable("asking for RRTI for a placeholder type!");

  case BuiltinType::ObjCId:
  case BuiltinType::ObjCClass:
  case BuiltinType::ObjCSel:
    llvm_unreachable("FIXME: Objective-C types are unsupported!");
  }

  llvm_unreachable("Invalid BuiltinType Kind!");
}

static bool typeInfoIsInStandardLibrary(const PointerType *pointerTy) {
  QualType pointeeTy = pointerTy->getPointeeType();
  const BuiltinType *builtinTy = dyn_cast<BuiltinType>(pointeeTy);
  if (!builtinTy)
    return false;

  // Check the qualifiers.
  Qualifiers quals = pointeeTy.getQualifiers();
  quals.removeConst();

  if (!quals.empty())
    return false;

  return typeInfoIsInStandardLibrary(builtinTy);
}

/// Returns whether the type
/// information for the given type exists in the standard library.
/// TODO(cir): this can unified with LLVM codegen
static bool isStandardLibraryRttiDescriptor(QualType ty) {
  // Type info for builtin types is defined in the standard library.
  if (const BuiltinType *builtinTy = dyn_cast<BuiltinType>(ty))
    return typeInfoIsInStandardLibrary(builtinTy);

  // Type info for some pointer types to builtin types is defined in the
  // standard library.
  if (const PointerType *pointerTy = dyn_cast<PointerType>(ty))
    return typeInfoIsInStandardLibrary(pointerTy);

  return false;
}

/// Returns whether the type information for
/// the given type exists somewhere else, and that we should not emit the type
/// information in this translation unit.  Assumes that it is not a
/// standard-library type.
/// TODO(cir): this can unified with LLVM codegen
static bool shouldUseExternalRttiDescriptor(CIRGenModule &cgm, QualType ty) {
  ASTContext &context = cgm.getASTContext();

  // If RTTI is disabled, assume it might be disabled in the
  // translation unit that defines any potential key function, too.
  if (!context.getLangOpts().RTTI)
    return false;

  if (const RecordType *recordTy = dyn_cast<RecordType>(ty)) {
    const CXXRecordDecl *rd = cast<CXXRecordDecl>(recordTy->getDecl());
    if (!rd->hasDefinition())
      return false;

    if (!rd->isDynamicClass())
      return false;

    // FIXME: this may need to be reconsidered if the key function
    // changes.
    // N.B. We must always emit the RTTI data ourselves if there exists a key
    // function.
    bool isDLLImport = rd->hasAttr<DLLImportAttr>();

    // Don't import the RTTI but emit it locally.
    if (cgm.getTriple().isWindowsGNUEnvironment())
      return false;

    if (cgm.getVTables().isVTableExternal(rd)) {
      if (cgm.getTarget().hasPS4DLLImportExport())
        return true;

      return !(isDLLImport && !cgm.getTriple().isWindowsItaniumEnvironment());
    }
    if (isDLLImport)
      return true;
  }

  return false;
}

/// Returns whether the given record type is incomplete.
/// TODO(cir): this can unified with LLVM codegen
static bool isIncompleteClassType(const RecordType *recordTy) {
  return !recordTy->getDecl()->isCompleteDefinition();
}

/// Returns whether the given type contains an
/// incomplete class type. This is true if
///
///   * The given type is an incomplete class type.
///   * The given type is a pointer type whose pointee type contains an
///     incomplete class type.
///   * The given type is a member pointer type whose class is an incomplete
///     class type.
///   * The given type is a member pointer type whoise pointee type contains an
///     incomplete class type.
/// is an indirect or direct pointer to an incomplete class type.
/// TODO(cir): this can unified with LLVM codegen
static bool containsIncompleteClassType(QualType ty) {
  if (const RecordType *recordTy = dyn_cast<RecordType>(ty)) {
    if (isIncompleteClassType(recordTy))
      return true;
  }

  if (const PointerType *pointerTy = dyn_cast<PointerType>(ty))
    return containsIncompleteClassType(pointerTy->getPointeeType());

  if (const MemberPointerType *memberPointerTy =
          dyn_cast<MemberPointerType>(ty)) {
    // Check if the class type is incomplete.
    const RecordType *classType = cast<RecordType>(memberPointerTy->getClass());
    if (isIncompleteClassType(classType))
      return true;

    return containsIncompleteClassType(memberPointerTy->getPointeeType());
  }

  return false;
}

// Return whether the given record decl has a "single,
// public, non-virtual base at offset zero (i.e. the derived class is dynamic
// iff the base is)", according to Itanium C++ ABI, 2.95p6b.
// TODO(cir): this can unified with LLVM codegen
static bool canUseSingleInheritance(const CXXRecordDecl *rd) {
  // Check the number of bases.
  if (rd->getNumBases() != 1)
    return false;

  // Get the base.
  CXXRecordDecl::base_class_const_iterator base = rd->bases_begin();

  // Check that the base is not virtual.
  if (base->isVirtual())
    return false;

  // Check that the base is public.
  if (base->getAccessSpecifier() != AS_public)
    return false;

  // Check that the class is dynamic iff the base is.
  auto *baseDecl =
      cast<CXXRecordDecl>(base->getType()->castAs<RecordType>()->getDecl());
  return !(!baseDecl->isEmpty() &&
           baseDecl->isDynamicClass() != rd->isDynamicClass());
}

/// Return the linkage that the type info and type info name constants
/// should have for the given type.
static mlir::cir::GlobalLinkageKind getTypeInfoLinkage(CIRGenModule &cgm,
                                                       QualType ty) {
  // Itanium C++ ABI 2.9.5p7:
  //   In addition, it and all of the intermediate abi::__pointer_type_info
  //   structs in the chain down to the abi::__class_type_info for the
  //   incomplete class type must be prevented from resolving to the
  //   corresponding type_info structs for the complete class type, possibly
  //   by making them local static objects. Finally, a dummy class RTTI is
  //   generated for the incomplete type that will not resolve to the final
  //   complete class RTTI (because the latter need not exist), possibly by
  //   making it a local static object.
  if (containsIncompleteClassType(ty))
    return mlir::cir::GlobalLinkageKind::InternalLinkage;

  switch (ty->getLinkage()) {
  case Linkage::None:
  case Linkage::Internal:
  case Linkage::UniqueExternal:
    return mlir::cir::GlobalLinkageKind::InternalLinkage;

  case Linkage::VisibleNone:
  case Linkage::Module:
  case Linkage::External:
    // RTTI is not enabled, which means that this type info struct is going
    // to be used for exception handling. Give it linkonce_odr linkage.
    if (!cgm.getLangOpts().RTTI)
      return mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage;

    if (const RecordType *record = dyn_cast<RecordType>(ty)) {
      const CXXRecordDecl *rd = cast<CXXRecordDecl>(record->getDecl());
      if (rd->hasAttr<WeakAttr>())
        return mlir::cir::GlobalLinkageKind::WeakODRLinkage;
      if (cgm.getTriple().isWindowsItaniumEnvironment())
        if (rd->hasAttr<DLLImportAttr>() &&
            shouldUseExternalRttiDescriptor(cgm, ty))
          return mlir::cir::GlobalLinkageKind::ExternalLinkage;
      // MinGW always uses LinkOnceODRLinkage for type info.
      if (rd->isDynamicClass() && !cgm.getASTContext()
                                       .getTargetInfo()
                                       .getTriple()
                                       .isWindowsGNUEnvironment())
        return cgm.getVTableLinkage(rd);
    }

    return mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage;
  case Linkage::Invalid:
    llvm_unreachable("Invalid linkage!");
  }

  llvm_unreachable("Invalid linkage!");
}

mlir::Attribute CIRGenItaniumRTTIBuilder::buildTypeInfo(mlir::Location loc,
                                                        QualType ty) {
  // We want to operate on the canonical type.
  ty = ty.getCanonicalType();

  // Check if we've already emitted an RTTI descriptor for this type.
  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTI(ty, out);

  auto oldGv = dyn_cast_or_null<mlir::cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(cgm.getModule(), name));

  if (oldGv && !oldGv.isDeclaration()) {
    assert(!oldGv.hasAvailableExternallyLinkage() &&
           "available_externally typeinfos not yet implemented");
    return cgm.getBuilder().getGlobalViewAttr(cgm.getBuilder().getUInt8PtrTy(),
                                              oldGv);
  }

  // Check if there is already an external RTTI descriptor for this type.
  if (isStandardLibraryRttiDescriptor(ty) ||
      shouldUseExternalRttiDescriptor(cgm, ty))
    return getAddrOfExternalRttiDescriptor(loc, ty);

  // Emit the standard library with external linkage.
  auto linkage = getTypeInfoLinkage(cgm, ty);

  // Give the type_info object and name the formal visibility of the
  // type itself.
  assert(!MissingFeatures::hiddenVisibility());
  assert(!MissingFeatures::protectedVisibility());
  mlir::SymbolTable::Visibility symVisibility;
  if (mlir::cir::isLocalLinkage(linkage))
    // If the linkage is local, only default visibility makes sense.
    symVisibility = mlir::SymbolTable::Visibility::Public;
  else if (cxxabi.classifyRTTIUniqueness(ty, linkage) ==
           CIRGenItaniumCXXABI::RUK_NonUniqueHidden)
    llvm_unreachable("NYI");
  else
    symVisibility = CIRGenModule::getCIRVisibility(ty->getVisibility());

  assert(!MissingFeatures::setDLLStorageClass());
  return buildTypeInfo(loc, ty, linkage, symVisibility);
}

void CIRGenItaniumRTTIBuilder::buildVTablePointer(mlir::Location loc,
                                                  const Type *ty) {
  auto &builder = cgm.getBuilder();

  // abi::__class_type_info.
  static const char *const classTypeInfo =
      "_ZTVN10__cxxabiv117__class_type_infoE";
  // abi::__si_class_type_info.
  static const char *const siClassTypeInfo =
      "_ZTVN10__cxxabiv120__si_class_type_infoE";
  // abi::__vmi_class_type_info.
  static const char *const vmiClassTypeInfo =
      "_ZTVN10__cxxabiv121__vmi_class_type_infoE";

  const char *vTableName = nullptr;

  switch (ty->getTypeClass()) {
  case Type::ArrayParameter:
    llvm_unreachable("NYI");
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Undeduced type shouldn't get here");

  case Type::Pipe:
    llvm_unreachable("Pipe types shouldn't get here");

  case Type::Builtin:
  case Type::BitInt:
  // GCC treats vector and complex types as fundamental types.
  case Type::Vector:
  case Type::ExtVector:
  case Type::ConstantMatrix:
  case Type::Complex:
  case Type::Atomic:
  // FIXME: GCC treats block pointers as fundamental types?!
  case Type::BlockPointer:
    // abi::__fundamental_type_info.
    vTableName = "_ZTVN10__cxxabiv123__fundamental_type_infoE";
    break;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    // abi::__array_type_info.
    vTableName = "_ZTVN10__cxxabiv117__array_type_infoE";
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // abi::__function_type_info.
    vTableName = "_ZTVN10__cxxabiv120__function_type_infoE";
    break;

  case Type::Enum:
    // abi::__enum_type_info.
    vTableName = "_ZTVN10__cxxabiv116__enum_type_infoE";
    break;

  case Type::Record: {
    const CXXRecordDecl *rd =
        cast<CXXRecordDecl>(cast<RecordType>(ty)->getDecl());

    if (!rd->hasDefinition() || !rd->getNumBases()) {
      vTableName = classTypeInfo;
    } else if (canUseSingleInheritance(rd)) {
      vTableName = siClassTypeInfo;
    } else {
      vTableName = vmiClassTypeInfo;
    }

    break;
  }

  case Type::ObjCObject:
    // Ignore protocol qualifiers.
    ty = cast<ObjCObjectType>(ty)->getBaseType().getTypePtr();

    // Handle id and Class.
    if (isa<BuiltinType>(ty)) {
      vTableName = classTypeInfo;
      break;
    }

    assert(isa<ObjCInterfaceType>(ty));
    [[fallthrough]];

  case Type::ObjCInterface:
    if (cast<ObjCInterfaceType>(ty)->getDecl()->getSuperClass()) {
      vTableName = siClassTypeInfo;
    } else {
      vTableName = classTypeInfo;
    }
    break;

  case Type::ObjCObjectPointer:
  case Type::Pointer:
    // abi::__pointer_type_info.
    vTableName = "_ZTVN10__cxxabiv119__pointer_type_infoE";
    break;

  case Type::MemberPointer:
    // abi::__pointer_to_member_type_info.
    vTableName = "_ZTVN10__cxxabiv129__pointer_to_member_type_infoE";
    break;
  }

  mlir::cir::GlobalOp vTable{};

  // Check if the alias exists. If it doesn't, then get or create the global.
  if (cgm.getItaniumVTableContext().isRelativeLayout())
    llvm_unreachable("NYI");
  if (!vTable) {
    vTable = cgm.getOrInsertGlobal(loc, vTableName,
                                   cgm.getBuilder().getUInt8PtrTy());
  }

  if (MissingFeatures::setDSOLocal())
    llvm_unreachable("NYI");

  // The vtable address point is 2.
  mlir::Attribute field{};
  if (cgm.getItaniumVTableContext().isRelativeLayout()) {
    llvm_unreachable("NYI");
  } else {
    SmallVector<mlir::Attribute, 4> offsets{
        cgm.getBuilder().getI32IntegerAttr(2)};
    auto indices = mlir::ArrayAttr::get(builder.getContext(), offsets);
    field = cgm.getBuilder().getGlobalViewAttr(cgm.getBuilder().getUInt8PtrTy(),
                                               vTable, indices);
  }

  assert(field && "expected attribute");
  fields.push_back(field);
}

mlir::cir::GlobalOp CIRGenItaniumRTTIBuilder::getAddrOfTypeName(
    mlir::Location loc, QualType ty, mlir::cir::GlobalLinkageKind linkage) {
  auto &builder = cgm.getBuilder();
  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTIName(ty, out);

  // We know that the mangled name of the type starts at index 4 of the
  // mangled name of the typename, so we can just index into it in order to
  // get the mangled name of the type.
  auto init = builder.getString(
      name.substr(4), cgm.getTypes().ConvertType(cgm.getASTContext().CharTy));
  auto align =
      cgm.getASTContext().getTypeAlignInChars(cgm.getASTContext().CharTy);

  // builder.getString can return a #cir.zero if the string given to it only
  // contains null bytes. However, type names cannot be full of null bytes.
  // So cast Init to a ConstArrayAttr should be safe.
  auto initStr = cast<mlir::cir::ConstArrayAttr>(init);

  auto gv = cgm.createOrReplaceCXXRuntimeVariable(loc, name, initStr.getType(),
                                                  linkage, align);
  CIRGenModule::setInitializer(gv, init);
  return gv;
}

/// Build an abi::__si_class_type_info, used for single inheritance, according
/// to the Itanium C++ ABI, 2.95p6b.
void CIRGenItaniumRTTIBuilder::buildSiClassTypeInfo(mlir::Location loc,
                                                    const CXXRecordDecl *rd) {
  // Itanium C++ ABI 2.9.5p6b:
  // It adds to abi::__class_type_info a single member pointing to the
  // type_info structure for the base type,
  auto baseTypeInfo = CIRGenItaniumRTTIBuilder(cxxabi, cgm)
                          .buildTypeInfo(loc, rd->bases_begin()->getType());
  fields.push_back(baseTypeInfo);
}

namespace {
/// Contains virtual and non-virtual bases seen when traversing a class
/// hierarchy.
struct SeenBases {
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> nonVirtualBases;
  llvm::SmallPtrSet<const CXXRecordDecl *, 16> virtualBases;
};
} // namespace

/// Compute the value of the flags member in abi::__vmi_class_type_info.
///
static unsigned computeVmiClassTypeInfoFlags(const CXXBaseSpecifier *base,
                                             SeenBases &bases) {

  unsigned flags = 0;

  auto *baseDecl =
      cast<CXXRecordDecl>(base->getType()->castAs<RecordType>()->getDecl());

  if (base->isVirtual()) {
    // Mark the virtual base as seen.
    if (!bases.virtualBases.insert(baseDecl).second) {
      // If this virtual base has been seen before, then the class is diamond
      // shaped.
      flags |= CIRGenItaniumRTTIBuilder::VMI_DiamondShaped;
    } else {
      if (bases.nonVirtualBases.count(baseDecl))
        flags |= CIRGenItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    }
  } else {
    // Mark the non-virtual base as seen.
    if (!bases.nonVirtualBases.insert(baseDecl).second) {
      // If this non-virtual base has been seen before, then the class has non-
      // diamond shaped repeated inheritance.
      flags |= CIRGenItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    } else {
      if (bases.virtualBases.count(baseDecl))
        flags |= CIRGenItaniumRTTIBuilder::VMI_NonDiamondRepeat;
    }
  }

  // Walk all bases.
  for (const auto &i : baseDecl->bases())
    flags |= computeVmiClassTypeInfoFlags(&i, bases);

  return flags;
}

static unsigned computeVmiClassTypeInfoFlags(const CXXRecordDecl *rd) {
  unsigned flags = 0;
  SeenBases bases;

  // Walk all bases.
  for (const auto &i : rd->bases())
    flags |= computeVmiClassTypeInfoFlags(&i, bases);

  return flags;
}

/// Build an abi::__vmi_class_type_info, used for
/// classes with bases that do not satisfy the abi::__si_class_type_info
/// constraints, according to the Itanium C++ ABI, 2.9.5p5c.
void CIRGenItaniumRTTIBuilder::buildVmiClassTypeInfo(mlir::Location loc,
                                                     const CXXRecordDecl *rd) {
  auto unsignedIntLTy =
      cgm.getTypes().ConvertType(cgm.getASTContext().UnsignedIntTy);
  // Itanium C++ ABI 2.9.5p6c:
  //   __flags is a word with flags describing details about the class
  //   structure, which may be referenced by using the __flags_masks
  //   enumeration. These flags refer to both direct and indirect bases.
  unsigned flags = computeVmiClassTypeInfoFlags(rd);
  fields.push_back(mlir::cir::IntAttr::get(unsignedIntLTy, flags));

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_count is a word with the number of direct proper base class
  //   descriptions that follow.
  fields.push_back(mlir::cir::IntAttr::get(unsignedIntLTy, rd->getNumBases()));

  if (!rd->getNumBases())
    return;

  // Now add the base class descriptions.

  // Itanium C++ ABI 2.9.5p6c:
  //   __base_info[] is an array of base class descriptions -- one for every
  //   direct proper base. Each description is of the type:
  //
  //   struct abi::__base_class_type_info {
  //   public:
  //     const __class_type_info *__base_type;
  //     long __offset_flags;
  //
  //     enum __offset_flags_masks {
  //       __virtual_mask = 0x1,
  //       __public_mask = 0x2,
  //       __offset_shift = 8
  //     };
  //   };

  // If we're in mingw and 'long' isn't wide enough for a pointer, use 'long
  // long' instead of 'long' for __offset_flags. libstdc++abi uses long long on
  // LLP64 platforms.
  // FIXME: Consider updating libc++abi to match, and extend this logic to all
  // LLP64 platforms.
  QualType offsetFlagsTy = cgm.getASTContext().LongTy;
  const TargetInfo &ti = cgm.getASTContext().getTargetInfo();
  if (ti.getTriple().isOSCygMing() &&
      ti.getPointerWidth(LangAS::Default) > ti.getLongWidth())
    offsetFlagsTy = cgm.getASTContext().LongLongTy;
  auto offsetFlagsLTy = cgm.getTypes().ConvertType(offsetFlagsTy);

  for (const auto &base : rd->bases()) {
    // The __base_type member points to the RTTI for the base type.
    fields.push_back(CIRGenItaniumRTTIBuilder(cxxabi, cgm)
                         .buildTypeInfo(loc, base.getType()));

    auto *baseDecl =
        cast<CXXRecordDecl>(base.getType()->castAs<RecordType>()->getDecl());

    int64_t offsetFlags = 0;

    // All but the lower 8 bits of __offset_flags are a signed offset.
    // For a non-virtual base, this is the offset in the object of the base
    // subobject. For a virtual base, this is the offset in the virtual table of
    // the virtual base offset for the virtual base referenced (negative).
    CharUnits offset;
    if (base.isVirtual())
      offset = cgm.getItaniumVTableContext().getVirtualBaseOffsetOffset(
          rd, baseDecl);
    else {
      const ASTRecordLayout &layout =
          cgm.getASTContext().getASTRecordLayout(rd);
      offset = layout.getBaseClassOffset(baseDecl);
    }
    offsetFlags = uint64_t(offset.getQuantity()) << 8;

    // The low-order byte of __offset_flags contains flags, as given by the
    // masks from the enumeration __offset_flags_masks.
    if (base.isVirtual())
      offsetFlags |= BCTI_Virtual;
    if (base.getAccessSpecifier() == AS_public)
      offsetFlags |= BCTI_Public;

    fields.push_back(mlir::cir::IntAttr::get(offsetFlagsLTy, offsetFlags));
  }
}

mlir::Attribute
CIRGenItaniumRTTIBuilder::getAddrOfExternalRttiDescriptor(mlir::Location loc,
                                                          QualType ty) {
  // Mangle the RTTI name.
  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTI(ty, out);
  auto &builder = cgm.getBuilder();

  // Look for an existing global.
  auto gv = dyn_cast_or_null<mlir::cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(cgm.getModule(), name));

  if (!gv) {
    // Create a new global variable.
    // From LLVM codegen => Note for the future: If we would ever like to do
    // deferred emission of RTTI, check if emitting vtables opportunistically
    // need any adjustment.
    gv = CIRGenModule::createGlobalOp(cgm, loc, name, builder.getUInt8PtrTy(),
                                      /*isConstant=*/true);
    const CXXRecordDecl *rd = ty->getAsCXXRecordDecl();
    cgm.setGVProperties(gv, rd);

    // Import the typeinfo symbol when all non-inline virtual methods are
    // imported.
    if (cgm.getTarget().hasPS4DLLImportExport())
      llvm_unreachable("NYI");
  }

  return builder.getGlobalViewAttr(builder.getUInt8PtrTy(), gv);
}

mlir::Attribute CIRGenItaniumRTTIBuilder::buildTypeInfo(
    mlir::Location loc, QualType ty, mlir::cir::GlobalLinkageKind linkage,
    mlir::SymbolTable::Visibility visibility) {
  auto &builder = cgm.getBuilder();
  assert(!MissingFeatures::setDLLStorageClass());

  // Add the vtable pointer.
  buildVTablePointer(loc, cast<Type>(ty));

  // And the name.
  auto typeName = getAddrOfTypeName(loc, ty, linkage);
  mlir::Attribute typeNameField;

  // If we're supposed to demote the visibility, be sure to set a flag
  // to use a string comparison for type_info comparisons.
  CIRGenItaniumCXXABI::RTTIUniquenessKind rttiUniqueness =
      cxxabi.classifyRTTIUniqueness(ty, linkage);
  if (rttiUniqueness != CIRGenItaniumCXXABI::RUK_Unique) {
    // The flag is the sign bit, which on ARM64 is defined to be clear
    // for global pointers.  This is very ARM64-specific.
    llvm_unreachable("NYI");
  } else {
    typeNameField =
        builder.getGlobalViewAttr(builder.getUInt8PtrTy(), typeName);
  }
  fields.push_back(typeNameField);

  switch (ty->getTypeClass()) {
  case Type::ArrayParameter:
    llvm_unreachable("NYI");
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.inc"
    llvm_unreachable("Non-canonical and dependent types shouldn't get here");

  // GCC treats vector types as fundamental types.
  case Type::Builtin:
  case Type::Vector:
  case Type::ExtVector:
  case Type::ConstantMatrix:
  case Type::Complex:
  case Type::BlockPointer:
    // Itanium C++ ABI 2.9.5p4:
    // abi::__fundamental_type_info adds no data members to std::type_info.
    break;

  case Type::LValueReference:
  case Type::RValueReference:
    llvm_unreachable("References shouldn't get here");

  case Type::Auto:
  case Type::DeducedTemplateSpecialization:
    llvm_unreachable("Undeduced type shouldn't get here");

  case Type::Pipe:
    break;

  case Type::BitInt:
    break;

  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__array_type_info adds no data members to std::type_info.
    break;

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__function_type_info adds no data members to std::type_info.
    break;

  case Type::Enum:
    // Itanium C++ ABI 2.9.5p5:
    // abi::__enum_type_info adds no data members to std::type_info.
    break;

  case Type::Record: {
    const CXXRecordDecl *rd =
        cast<CXXRecordDecl>(cast<RecordType>(ty)->getDecl());
    if (!rd->hasDefinition() || !rd->getNumBases()) {
      // We don't need to emit any fields.
      break;
    }

    if (canUseSingleInheritance(rd)) {
      buildSiClassTypeInfo(loc, rd);
    } else {
      buildVmiClassTypeInfo(loc, rd);
    }

    break;
  }

  case Type::ObjCObject:
  case Type::ObjCInterface:
    llvm_unreachable("NYI");
    break;

  case Type::ObjCObjectPointer:
    llvm_unreachable("NYI");
    break;

  case Type::Pointer:
    llvm_unreachable("NYI");
    break;

  case Type::MemberPointer:
    llvm_unreachable("NYI");
    break;

  case Type::Atomic:
    // No fields, at least for the moment.
    break;
  }

  assert(!MissingFeatures::setDLLImportDLLExport());
  auto init = builder.getTypeInfo(builder.getArrayAttr(fields));

  SmallString<256> name;
  llvm::raw_svector_ostream out(name);
  cgm.getCXXABI().getMangleContext().mangleCXXRTTI(ty, out);

  // Create new global and search for an existing global.
  auto oldGv = dyn_cast_or_null<mlir::cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(cgm.getModule(), name));
  mlir::cir::GlobalOp gv =
      CIRGenModule::createGlobalOp(cgm, loc, name, init.getType(),
                                   /*isConstant=*/true);

  // Export the typeinfo in the same circumstances as the vtable is
  // exported.
  if (cgm.getTarget().hasPS4DLLImportExport())
    llvm_unreachable("NYI");

  // If there's already an old global variable, replace it with the new one.
  if (oldGv) {
    // Replace occurrences of the old variable if needed.
    gv.setName(oldGv.getName());
    if (!oldGv->use_empty()) {
      // TODO: replaceAllUsesWith
      llvm_unreachable("NYI");
    }
    oldGv->erase();
  }

  if (cgm.supportsCOMDAT() && mlir::cir::isWeakForLinker(gv.getLinkage())) {
    assert(!MissingFeatures::setComdat());
    llvm_unreachable("NYI");
  }

  CharUnits align = cgm.getASTContext().toCharUnitsFromBits(
      cgm.getTarget().getPointerAlign(LangAS::Default));
  gv.setAlignmentAttr(cgm.getSize(align));

  // The Itanium ABI specifies that type_info objects must be globally
  // unique, with one exception: if the type is an incomplete class
  // type or a (possibly indirect) pointer to one.  That exception
  // affects the general case of comparing type_info objects produced
  // by the typeid operator, which is why the comparison operators on
  // std::type_info generally use the type_info name pointers instead
  // of the object addresses.  However, the language's built-in uses
  // of RTTI generally require class types to be complete, even when
  // manipulating pointers to those class types.  This allows the
  // implementation of dynamic_cast to rely on address equality tests,
  // which is much faster.
  //
  // All of this is to say that it's important that both the type_info
  // object and the type_info name be uniqued when weakly emitted.

  // TODO(cir): setup other bits for TypeName
  assert(!MissingFeatures::setDLLStorageClass());
  assert(!MissingFeatures::setPartition());
  assert(!MissingFeatures::setDSOLocal());
  mlir::SymbolTable::setSymbolVisibility(
      typeName, CIRGenModule::getMLIRVisibility(typeName));

  // TODO(cir): setup other bits for GV
  assert(!MissingFeatures::setDLLStorageClass());
  assert(!MissingFeatures::setPartition());
  assert(!MissingFeatures::setDSOLocal());
  CIRGenModule::setInitializer(gv, init);

  return builder.getGlobalViewAttr(builder.getUInt8PtrTy(), gv);
  ;
}

mlir::Attribute CIRGenItaniumCXXABI::getAddrOfRTTIDescriptor(mlir::Location loc,
                                                             QualType ty) {
  return CIRGenItaniumRTTIBuilder(*this, CGM).buildTypeInfo(loc, ty);
}

void CIRGenItaniumCXXABI::emitVTableDefinitions(CIRGenVTables &cgvt,
                                                const CXXRecordDecl *rd) {
  auto vTable = getAddrOfVTable(rd, CharUnits());
  if (vTable.hasInitializer())
    return;

  ItaniumVTableContext &vtContext = CGM.getItaniumVTableContext();
  const VTableLayout &vtLayout = vtContext.getVTableLayout(rd);
  auto linkage = CGM.getVTableLinkage(rd);
  auto rtti = CGM.getAddrOfRTTIDescriptor(
      CGM.getLoc(rd->getBeginLoc()), CGM.getASTContext().getTagDeclType(rd));

  // Create and set the initializer.
  ConstantInitBuilder builder(CGM);
  auto components = builder.beginStruct();

  cgvt.createVTableInitializer(components, vtLayout, rtti,
                               mlir::cir::isLocalLinkage(linkage));
  components.finishAndSetAsInitializer(vTable, /*forVtable=*/true);

  // Set the correct linkage.
  vTable.setLinkage(linkage);

  if (CGM.supportsCOMDAT() && mlir::cir::isWeakForLinker(linkage)) {
    assert(!MissingFeatures::setComdat());
  }

  // Set the right visibility.
  CGM.setGVProperties(vTable, rd);

  // If this is the magic class __cxxabiv1::__fundamental_type_info,
  // we will emit the typeinfo for the fundamental types. This is the
  // same behaviour as GCC.
  const DeclContext *dc = rd->getDeclContext();
  if (rd->getIdentifier() &&
      rd->getIdentifier()->isStr("__fundamental_type_info") &&
      isa<NamespaceDecl>(dc) && cast<NamespaceDecl>(dc)->getIdentifier() &&
      cast<NamespaceDecl>(dc)->getIdentifier()->isStr("__cxxabiv1") &&
      dc->getParent()->isTranslationUnit()) {
    llvm_unreachable("NYI");
    // EmitFundamentalRTTIDescriptors(RD);
  }

  auto vTableAsGlobalValue =
      dyn_cast<mlir::cir::CIRGlobalValueInterface>(*vTable);
  assert(vTableAsGlobalValue && "VTable must support CIRGlobalValueInterface");
  bool isDeclarationForLinker = vTableAsGlobalValue.isDeclarationForLinker();
  // Always emit type metadata on non-available_externally definitions, and on
  // available_externally definitions if we are performing whole program
  // devirtualization. For WPD we need the type metadata on all vtable
  // definitions to ensure we associate derived classes with base classes
  // defined in headers but with a strong definition only in a shared
  // library.
  if (!isDeclarationForLinker || CGM.getCodeGenOpts().WholeProgramVTables) {
    CGM.buildVTableTypeMetadata(rd, vTable, vtLayout);
    // For available_externally definitions, add the vtable to
    // @llvm.compiler.used so that it isn't deleted before whole program
    // analysis.
    if (isDeclarationForLinker) {
      llvm_unreachable("NYI");
      assert(CGM.getCodeGenOpts().WholeProgramVTables);
      assert(!MissingFeatures::addCompilerUsedGlobal());
    }
  }

  if (vtContext.isRelativeLayout())
    llvm_unreachable("NYI");
}

void CIRGenItaniumCXXABI::emitVirtualInheritanceTables(
    const CXXRecordDecl *rd) {
  CIRGenVTables &vTables = CGM.getVTables();
  auto vtt = vTables.getAddrOfVTT(rd);
  vTables.buildVTTDefinition(vtt, CGM.getVTableLinkage(rd), rd);
}

/// What sort of uniqueness rules should we use for the RTTI for the
/// given type?
CIRGenItaniumCXXABI::RTTIUniquenessKind
CIRGenItaniumCXXABI::classifyRTTIUniqueness(
    QualType canTy, mlir::cir::GlobalLinkageKind linkage) const {
  if (shouldRTTIBeUnique())
    return RUK_Unique;

  // It's only necessary for linkonce_odr or weak_odr linkage.
  if (linkage != mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage &&
      linkage != mlir::cir::GlobalLinkageKind::WeakODRLinkage)
    return RUK_Unique;

  // It's only necessary with default visibility.
  if (canTy->getVisibility() != DefaultVisibility)
    return RUK_Unique;

  // If we're not required to publish this symbol, hide it.
  if (linkage == mlir::cir::GlobalLinkageKind::LinkOnceODRLinkage)
    return RUK_NonUniqueHidden;

  // If we're required to publish this symbol, as we might be under an
  // explicit instantiation, leave it with default visibility but
  // enable string-comparisons.
  assert(linkage == mlir::cir::GlobalLinkageKind::WeakODRLinkage);
  return RUK_NonUniqueVisible;
}

void CIRGenItaniumCXXABI::buildDestructorCall(
    CIRGenFunction &cgf, const CXXDestructorDecl *dd, CXXDtorType type,
    bool forVirtualBase, bool delegating, Address theThis, QualType thisTy) {
  GlobalDecl gd(dd, type);
  auto vtt =
      getCXXDestructorImplicitParam(cgf, dd, type, forVirtualBase, delegating);
  QualType vttTy = getContext().getPointerType(getContext().VoidPtrTy);
  CIRGenCallee callee;
  if (getContext().getLangOpts().AppleKext && type != Dtor_Base &&
      dd->isVirtual())
    llvm_unreachable("NYI");
  else
    callee = CIRGenCallee::forDirect(CGM.getAddrOfCXXStructor(gd), gd);

  cgf.buildCXXDestructorCall(gd, callee, theThis.getPointer(), thisTy, vtt,
                             vttTy, nullptr);
}

void CIRGenItaniumCXXABI::registerGlobalDtor(CIRGenFunction &cgf,
                                             const VarDecl *d,
                                             mlir::cir::FuncOp dtor,
                                             mlir::Attribute addr) {
  if (d->isNoDestroy(CGM.getASTContext()))
    return;

  if (d->getTLSKind())
    llvm_unreachable("NYI");

  // HLSL doesn't support atexit.
  if (CGM.getLangOpts().HLSL)
    llvm_unreachable("NYI");

  // The default behavior is to use atexit. This is handled in lowering
  // prepare. Nothing to be done for CIR here.
}

mlir::Value CIRGenItaniumCXXABI::getCXXDestructorImplicitParam(
    CIRGenFunction &cgf, const CXXDestructorDecl *dd, CXXDtorType type,
    bool forVirtualBase, bool delegating) {
  GlobalDecl gd(dd, type);
  return cgf.GetVTTParameter(gd, forVirtualBase, delegating);
}

void CIRGenItaniumCXXABI::buildRethrow(CIRGenFunction &cgf, bool isNoReturn) {
  // void __cxa_rethrow();
  llvm_unreachable("NYI");
}

void CIRGenItaniumCXXABI::buildThrow(CIRGenFunction &cgf,
                                     const CXXThrowExpr *e) {
  // This differs a bit from LLVM codegen, CIR has native operations for some
  // cxa functions, and defers allocation size computation, always pass the dtor
  // symbol, etc. CIRGen also does not use getAllocateExceptionFn / getThrowFn.

  // Now allocate the exception object.
  auto &builder = cgf.getBuilder();
  QualType clangThrowType = e->getSubExpr()->getType();
  auto throwTy = builder.getPointerTo(cgf.ConvertType(clangThrowType));
  uint64_t typeSize =
      cgf.getContext().getTypeSizeInChars(clangThrowType).getQuantity();
  auto subExprLoc = cgf.getLoc(e->getSubExpr()->getSourceRange());
  // Defer computing allocation size to some later lowering pass.
  auto exceptionPtr =
      builder
          .create<mlir::cir::AllocExceptionOp>(
              subExprLoc, throwTy, builder.getI64IntegerAttr(typeSize))
          .getAddr();

  // Build expression and store its result into exceptionPtr.
  CharUnits exnAlign = cgf.getContext().getExnObjectAlignment();
  cgf.buildAnyExprToExn(e->getSubExpr(), Address(exceptionPtr, exnAlign));

  // Get the RTTI symbol address.
  auto typeInfo = mlir::dyn_cast_if_present<mlir::cir::GlobalViewAttr>(
      CGM.getAddrOfRTTIDescriptor(subExprLoc, clangThrowType,
                                  /*ForEH=*/true));
  assert(typeInfo && "expected GlobalViewAttr typeinfo");
  assert(!typeInfo.getIndices() && "expected no indirection");

  // The address of the destructor.
  //
  // Note: LLVM codegen already optimizes out the dtor if the
  // type is a record with trivial dtor (by passing down a
  // null dtor). In CIR, we forward this info and allow for
  // LoweringPrepare or some other pass to skip passing the
  // trivial function.
  //
  // TODO(cir): alternatively, dtor could be ignored here and
  // the type used to gather the relevant dtor during
  // LoweringPrepare.
  mlir::FlatSymbolRefAttr dtor{};
  if (const RecordType *recordTy = clangThrowType->getAs<RecordType>()) {
    CXXRecordDecl *rec = cast<CXXRecordDecl>(recordTy->getDecl());
    CXXDestructorDecl *dtorD = rec->getDestructor();
    dtor = mlir::FlatSymbolRefAttr::get(
        CGM.getAddrOfCXXStructor(GlobalDecl(dtorD, Dtor_Complete))
            .getSymNameAttr());
  }

  // FIXME: When adding support for invoking, we should wrap the throw op
  // below into a try, and let CFG flatten pass to generate a cir.try_call.
  assert(!cgf.isInvokeDest() && "landing pad like logic NYI");

  // Now throw the exception.
  mlir::Location loc = cgf.getLoc(e->getSourceRange());
  builder.create<mlir::cir::ThrowOp>(loc, exceptionPtr, typeInfo.getSymbol(),
                                     dtor);
  builder.create<mlir::cir::UnreachableOp>(loc);
}

static mlir::cir::FuncOp getBadCastFn(CIRGenFunction &cgf) {
  // Prototype: void __cxa_bad_cast();

  // TODO(cir): set the calling convention of the runtime function.
  assert(!MissingFeatures::setCallingConv());

  mlir::cir::FuncType fTy =
      cgf.getBuilder().getFuncType({}, cgf.getBuilder().getVoidTy());
  return cgf.cgm.createRuntimeFunction(fTy, "__cxa_bad_cast");
}

static void buildCallToBadCast(CIRGenFunction &cgf, mlir::Location loc) {
  // TODO(cir): set the calling convention to the runtime function.
  assert(!MissingFeatures::setCallingConv());

  cgf.buildRuntimeCall(loc, getBadCastFn(cgf));
  cgf.getBuilder().create<mlir::cir::UnreachableOp>(loc);
  cgf.getBuilder().clearInsertionPoint();
}

void CIRGenItaniumCXXABI::buildBadCastCall(CIRGenFunction &cgf,
                                           mlir::Location loc) {
  buildCallToBadCast(cgf, loc);
}

static CharUnits computeOffsetHint(ASTContext &context,
                                   const CXXRecordDecl *src,
                                   const CXXRecordDecl *dst) {
  CXXBasePaths paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);

  // If Dst is not derived from Src we can skip the whole computation below and
  // return that Src is not a public base of Dst.  Record all inheritance paths.
  if (!dst->isDerivedFrom(src, paths))
    return CharUnits::fromQuantity(-2ULL);

  unsigned numPublicPaths = 0;
  CharUnits offset;

  // Now walk all possible inheritance paths.
  for (const CXXBasePath &path : paths) {
    if (path.Access != AS_public) // Ignore non-public inheritance.
      continue;

    ++numPublicPaths;

    for (const CXXBasePathElement &pathElement : path) {
      // If the path contains a virtual base class we can't give any hint.
      // -1: no hint.
      if (pathElement.Base->isVirtual())
        return CharUnits::fromQuantity(-1ULL);

      if (numPublicPaths > 1) // Won't use offsets, skip computation.
        continue;

      // Accumulate the base class offsets.
      const ASTRecordLayout &l = context.getASTRecordLayout(pathElement.Class);
      offset += l.getBaseClassOffset(
          pathElement.Base->getType()->getAsCXXRecordDecl());
    }
  }

  // -2: Src is not a public base of Dst.
  if (numPublicPaths == 0)
    return CharUnits::fromQuantity(-2ULL);

  // -3: Src is a multiple public base type but never a virtual base type.
  if (numPublicPaths > 1)
    return CharUnits::fromQuantity(-3ULL);

  // Otherwise, the Src type is a unique public nonvirtual base type of Dst.
  // Return the offset of Src from the origin of Dst.
  return offset;
}

static mlir::cir::FuncOp getItaniumDynamicCastFn(CIRGenFunction &cgf) {
  // Prototype:
  // void *__dynamic_cast(const void *sub,
  //                      global_as const abi::__class_type_info *src,
  //                      global_as const abi::__class_type_info *dst,
  //                      std::ptrdiff_t src2dst_offset);

  mlir::Type voidPtrTy = cgf.VoidPtrTy;
  mlir::Type rttiPtrTy = cgf.getBuilder().getUInt8PtrTy();
  mlir::Type ptrDiffTy = cgf.ConvertType(cgf.getContext().getPointerDiffType());

  // TODO(cir): mark the function as nowind readonly.

  // TODO(cir): set the calling convention of the runtime function.
  assert(!MissingFeatures::setCallingConv());

  mlir::cir::FuncType fTy = cgf.getBuilder().getFuncType(
      {voidPtrTy, rttiPtrTy, rttiPtrTy, ptrDiffTy}, voidPtrTy);
  return cgf.cgm.createRuntimeFunction(fTy, "__dynamic_cast");
}

static Address buildDynamicCastToVoid(CIRGenFunction &cgf, mlir::Location loc,
                                      QualType srcRecordTy, Address src) {
  auto vtableUsesRelativeLayout =
      cgf.cgm.getItaniumVTableContext().isRelativeLayout();
  auto ptr = cgf.getBuilder().createDynCastToVoid(loc, src.getPointer(),
                                                  vtableUsesRelativeLayout);
  return Address{ptr, src.getAlignment()};
}

static mlir::Value
buildExactDynamicCast(CIRGenItaniumCXXABI &abi, CIRGenFunction &cgf,
                      mlir::Location loc, QualType srcRecordTy,
                      QualType destRecordTy, mlir::cir::PointerType destCirTy,
                      bool isRefCast, Address src) {
  // Find all the inheritance paths from SrcRecordTy to DestRecordTy.
  const CXXRecordDecl *srcDecl = srcRecordTy->getAsCXXRecordDecl();
  const CXXRecordDecl *destDecl = destRecordTy->getAsCXXRecordDecl();
  CXXBasePaths paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                     /*DetectVirtual=*/false);
  (void)destDecl->isDerivedFrom(srcDecl, paths);

  // Find an offset within `DestDecl` where a `SrcDecl` instance and its vptr
  // might appear.
  std::optional<CharUnits> offset;
  for (const CXXBasePath &path : paths) {
    // dynamic_cast only finds public inheritance paths.
    if (path.Access != AS_public)
      continue;

    CharUnits pathOffset;
    for (const CXXBasePathElement &pathElement : path) {
      // Find the offset along this inheritance step.
      const CXXRecordDecl *base =
          pathElement.Base->getType()->getAsCXXRecordDecl();
      if (pathElement.Base->isVirtual()) {
        // For a virtual base class, we know that the derived class is exactly
        // DestDecl, so we can use the vbase offset from its layout.
        const ASTRecordLayout &l =
            cgf.getContext().getASTRecordLayout(destDecl);
        pathOffset = l.getVBaseClassOffset(base);
      } else {
        const ASTRecordLayout &l =
            cgf.getContext().getASTRecordLayout(pathElement.Class);
        pathOffset += l.getBaseClassOffset(base);
      }
    }

    if (!offset)
      offset = pathOffset;
    else if (offset != pathOffset) {
      // Base appears in at least two different places. Find the most-derived
      // object and see if it's a DestDecl. Note that the most-derived object
      // must be at least as aligned as this base class subobject, and must
      // have a vptr at offset 0.
      src = buildDynamicCastToVoid(cgf, loc, srcRecordTy, src);
      srcDecl = destDecl;
      offset = CharUnits::Zero();
      break;
    }
  }

  if (!offset) {
    // If there are no public inheritance paths, the cast always fails.
    mlir::Value nullPtrValue = cgf.getBuilder().getNullPtr(destCirTy, loc);
    if (isRefCast) {
      auto *currentRegion = cgf.getBuilder().getBlock()->getParent();
      buildCallToBadCast(cgf, loc);

      // The call to bad_cast will terminate the block. Create a new block to
      // hold any follow up code.
      cgf.getBuilder().createBlock(currentRegion, currentRegion->end());
    }

    return nullPtrValue;
  }

  // Compare the vptr against the expected vptr for the destination type at
  // this offset. Note that we do not know what type Src points to in the case
  // where the derived class multiply inherits from the base class so we can't
  // use GetVTablePtr, so we load the vptr directly instead.

  mlir::Value expectedVPtr =
      abi.getVTableAddressPoint(BaseSubobject(srcDecl, *offset), destDecl);

  // TODO(cir): handle address space here.
  assert(!MissingFeatures::addressSpace());
  mlir::Type vPtrTy = expectedVPtr.getType();
  mlir::Type vPtrPtrTy = cgf.getBuilder().getPointerTo(vPtrTy);
  Address srcVPtrPtr(
      cgf.getBuilder().createBitcast(src.getPointer(), vPtrPtrTy),
      src.getAlignment());
  mlir::Value srcVPtr = cgf.getBuilder().createLoad(loc, srcVPtrPtr);

  // TODO(cir): decorate SrcVPtr with TBAA info.
  assert(!MissingFeatures::tbaa());

  mlir::Value success = cgf.getBuilder().createCompare(
      loc, mlir::cir::CmpOpKind::eq, srcVPtr, expectedVPtr);

  auto buildCastResult = [&] {
    if (offset->isZero())
      return cgf.getBuilder().createBitcast(src.getPointer(), destCirTy);

    // TODO(cir): handle address space here.
    assert(!MissingFeatures::addressSpace());
    mlir::Type u8PtrTy =
        cgf.getBuilder().getPointerTo(cgf.getBuilder().getUInt8Ty());

    mlir::Value strideToApply = cgf.getBuilder().getConstInt(
        loc, cgf.getBuilder().getUInt64Ty(), offset->getQuantity());
    mlir::Value srcU8Ptr =
        cgf.getBuilder().createBitcast(src.getPointer(), u8PtrTy);
    mlir::Value resultU8Ptr = cgf.getBuilder().create<mlir::cir::PtrStrideOp>(
        loc, u8PtrTy, srcU8Ptr, strideToApply);
    return cgf.getBuilder().createBitcast(resultU8Ptr, destCirTy);
  };

  if (isRefCast) {
    mlir::Value failed = cgf.getBuilder().createNot(success);
    cgf.getBuilder().create<mlir::cir::IfOp>(
        loc, failed, /*withElseRegion=*/false,
        [&](mlir::OpBuilder &, mlir::Location) {
          buildCallToBadCast(cgf, loc);
        });
    return buildCastResult();
  }

  return cgf.getBuilder()
      .create<mlir::cir::TernaryOp>(
          loc, success,
          [&](mlir::OpBuilder &, mlir::Location) {
            auto result = buildCastResult();
            cgf.getBuilder().createYield(loc, result);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value nullPtrValue =
                cgf.getBuilder().getNullPtr(destCirTy, loc);
            cgf.getBuilder().createYield(loc, nullPtrValue);
          })
      .getResult();
}

static mlir::cir::DynamicCastInfoAttr
buildDynamicCastInfo(CIRGenFunction &cgf, mlir::Location loc,
                     QualType srcRecordTy, QualType destRecordTy) {
  auto srcRtti = mlir::cast<mlir::cir::GlobalViewAttr>(
      cgf.cgm.getAddrOfRTTIDescriptor(loc, srcRecordTy));
  auto destRtti = mlir::cast<mlir::cir::GlobalViewAttr>(
      cgf.cgm.getAddrOfRTTIDescriptor(loc, destRecordTy));

  auto runtimeFuncOp = getItaniumDynamicCastFn(cgf);
  auto badCastFuncOp = getBadCastFn(cgf);
  auto runtimeFuncRef = mlir::FlatSymbolRefAttr::get(runtimeFuncOp);
  auto badCastFuncRef = mlir::FlatSymbolRefAttr::get(badCastFuncOp);

  const CXXRecordDecl *srcDecl = srcRecordTy->getAsCXXRecordDecl();
  const CXXRecordDecl *destDecl = destRecordTy->getAsCXXRecordDecl();
  auto offsetHint = computeOffsetHint(cgf.getContext(), srcDecl, destDecl);

  mlir::Type ptrdiffTy = cgf.ConvertType(cgf.getContext().getPointerDiffType());
  auto offsetHintAttr =
      mlir::cir::IntAttr::get(ptrdiffTy, offsetHint.getQuantity());

  return mlir::cir::DynamicCastInfoAttr::get(srcRtti, destRtti, runtimeFuncRef,
                                             badCastFuncRef, offsetHintAttr);
}

mlir::Value CIRGenItaniumCXXABI::buildDynamicCast(
    CIRGenFunction &cgf, mlir::Location loc, QualType srcRecordTy,
    QualType destRecordTy, mlir::cir::PointerType destCirTy, bool isRefCast,
    Address src) {
  bool isCastToVoid = destRecordTy.isNull();
  assert((!isCastToVoid || !isRefCast) && "cannot cast to void reference");

  if (isCastToVoid)
    return buildDynamicCastToVoid(cgf, loc, srcRecordTy, src).getPointer();

  // If the destination is effectively final, the cast succeeds if and only
  // if the dynamic type of the pointer is exactly the destination type.
  if (destRecordTy->getAsCXXRecordDecl()->isEffectivelyFinal() &&
      cgf.cgm.getCodeGenOpts().OptimizationLevel > 0)
    return buildExactDynamicCast(*this, cgf, loc, srcRecordTy, destRecordTy,
                                 destCirTy, isRefCast, src);

  auto castInfo = buildDynamicCastInfo(cgf, loc, srcRecordTy, destRecordTy);
  return cgf.getBuilder().createDynCast(loc, src.getPointer(), destCirTy,
                                        isRefCast, castInfo);
}

mlir::cir::MethodAttr
CIRGenItaniumCXXABI::buildVirtualMethodAttr(mlir::cir::MethodType methodTy,
                                            const CXXMethodDecl *md) {
  assert(md->isVirtual() && "only deal with virtual member functions");

  uint64_t index = CGM.getItaniumVTableContext().getMethodVTableIndex(md);
  uint64_t vTableOffset;
  if (CGM.getItaniumVTableContext().isRelativeLayout()) {
    // Multiply by 4-byte relative offsets.
    vTableOffset = index * 4;
  } else {
    const ASTContext &context = getContext();
    CharUnits pointerWidth = context.toCharUnitsFromBits(
        context.getTargetInfo().getPointerWidth(LangAS::Default));
    vTableOffset = index * pointerWidth.getQuantity();
  }

  return mlir::cir::MethodAttr::get(methodTy, vTableOffset);
}
