//===--- CIRGenClass.cpp - Emit CIR Code for C++ classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of classes
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace cir;

/// Checks whether the given constructor is a valid subject for the
/// complete-to-base constructor delgation optimization, i.e. emitting the
/// complete constructor as a simple call to the base constructor.
bool CIRGenFunction::IsConstructorDelegationValid(
    const CXXConstructorDecl *ctor) {

  // Currently we disable the optimization for classes with virtual bases
  // because (1) the address of parameter variables need to be consistent across
  // all initializers but (2) the delegate function call necessarily creates a
  // second copy of the parameter variable.
  //
  // The limiting example (purely theoretical AFAIK):
  //   struct A { A(int &c) { c++; } };
  //   struct A : virtual A {
  //     B(int count) : A(count) { printf("%d\n", count); }
  //   };
  // ...although even this example could in principle be emitted as a delegation
  // since the address of the parameter doesn't escape.
  if (ctor->getParent()->getNumVBases())
    return false;

  // We also disable the optimization for variadic functions because it's
  // impossible to "re-pass" varargs.
  if (ctor->getType()->castAs<FunctionProtoType>()->isVariadic())
    return false;

  // FIXME: Decide if we can do a delegation of a delegating constructor.
  if (ctor->isDelegatingConstructor())
    return false;

  return true;
}

/// TODO(cir): strong candidate for AST helper to be shared between LLVM and CIR
/// codegen.
static bool isMemcpyEquivalentSpecialMember(const CXXMethodDecl *d) {
  auto *cd = dyn_cast<CXXConstructorDecl>(d);
  if (!(cd && cd->isCopyOrMoveConstructor()) &&
      !d->isCopyAssignmentOperator() && !d->isMoveAssignmentOperator())
    return false;

  // We can emit a memcpy for a trivial copy or move constructor/assignment.
  if (d->isTrivial() && !d->getParent()->mayInsertExtraPadding())
    return true;

  // We *must* emit a memcpy for a defaulted union copy or move op.
  if (d->getParent()->isUnion() && d->isDefaulted())
    return true;

  return false;
}

namespace {
/// TODO(cir): a lot of what we see under this namespace is a strong candidate
/// to be shared between LLVM and CIR codegen.

/// RAII object to indicate that codegen is copying the value representation
/// instead of the object representation. Useful when copying a struct or
/// class which has uninitialized members and we're only performing
/// lvalue-to-rvalue conversion on the object but not its members.
class CopyingValueRepresentation {
public:
  explicit CopyingValueRepresentation(CIRGenFunction &cgf)
      : cgf(cgf), oldSanOpts(cgf.SanOpts) {
    cgf.SanOpts.set(SanitizerKind::Bool, false);
    cgf.SanOpts.set(SanitizerKind::Enum, false);
  }
  ~CopyingValueRepresentation() { cgf.SanOpts = oldSanOpts; }

private:
  CIRGenFunction &cgf;
  SanitizerSet oldSanOpts;
};

class FieldMemcpyizer {
public:
  FieldMemcpyizer(CIRGenFunction &cgf, const CXXRecordDecl *classDecl,
                  const VarDecl *srcRec)
      : cgf(cgf), classDecl(classDecl),
        // SrcRec(SrcRec),
        recLayout(cgf.getContext().getASTRecordLayout(classDecl)) {
    (void)srcRec;
  }

  bool isMemcpyableField(FieldDecl *f) const {
    // Never memcpy fields when we are adding poised paddings.
    if (cgf.getContext().getLangOpts().SanitizeAddressFieldPadding)
      return false;
    Qualifiers Qual = f->getType().getQualifiers();
    return !(Qual.hasVolatile() || Qual.hasObjCLifetime());
  }

  void addMemcpyableField(FieldDecl *f) {
    if (f->isZeroSize(cgf.getContext()))
      return;
    if (!FirstField)
      addInitialField(f);
    else
      addNextField(f);
  }

  CharUnits getMemcpySize(uint64_t firstByteOffset) const {
    ASTContext &ctx = cgf.getContext();
    unsigned lastFieldSize =
        LastField->isBitField()
            ? LastField->getBitWidthValue(ctx)
            : ctx.toBits(
                  ctx.getTypeInfoDataSizeInChars(LastField->getType()).Width);
    uint64_t memcpySizeBits = LastFieldOffset + lastFieldSize -
                              firstByteOffset + ctx.getCharWidth() - 1;
    CharUnits memcpySize = ctx.toCharUnitsFromBits(memcpySizeBits);
    return memcpySize;
  }

  void buildMemcpy() {
    // Give the subclass a chance to bail out if it feels the memcpy isn't worth
    // it (e.g. Hasn't aggregated enough data).
    if (!FirstField) {
      return;
    }

    llvm_unreachable("NYI");
  }

  void reset() { FirstField = nullptr; }

protected:
  CIRGenFunction &cgf;
  const CXXRecordDecl *classDecl;

private:
  void buildMemcpyIR(Address destPtr, Address srcPtr, CharUnits size) {
    llvm_unreachable("NYI");
  }

  void addInitialField(FieldDecl *f) {
    FirstField = f;
    LastField = f;
    FirstFieldOffset = recLayout.getFieldOffset(f->getFieldIndex());
    LastFieldOffset = FirstFieldOffset;
    LastAddedFieldIndex = f->getFieldIndex();
  }

  void addNextField(FieldDecl *f) {
    // For the most part, the following invariant will hold:
    //   F->getFieldIndex() == LastAddedFieldIndex + 1
    // The one exception is that Sema won't add a copy-initializer for an
    // unnamed bitfield, which will show up here as a gap in the sequence.
    assert(f->getFieldIndex() >= LastAddedFieldIndex + 1 &&
           "Cannot aggregate fields out of order.");
    LastAddedFieldIndex = f->getFieldIndex();

    // The 'first' and 'last' fields are chosen by offset, rather than field
    // index. This allows the code to support bitfields, as well as regular
    // fields.
    uint64_t fOffset = recLayout.getFieldOffset(f->getFieldIndex());
    if (fOffset < FirstFieldOffset) {
      FirstField = f;
      FirstFieldOffset = fOffset;
    } else if (fOffset >= LastFieldOffset) {
      LastField = f;
      LastFieldOffset = fOffset;
    }
  }

  // const VarDecl *SrcRec;
  const ASTRecordLayout &recLayout;
  FieldDecl *FirstField = nullptr;
  FieldDecl *LastField = nullptr;
  uint64_t FirstFieldOffset = 0, LastFieldOffset = 0;
  unsigned LastAddedFieldIndex = 0;
};

static void buildLValueForAnyFieldInitialization(CIRGenFunction &cgf,
                                                 CXXCtorInitializer *memberInit,
                                                 LValue &lhs) {
  FieldDecl *field = memberInit->getAnyMember();
  if (memberInit->isIndirectMemberInitializer()) {
    llvm_unreachable("NYI");
  } else {
    lhs = cgf.buildLValueForFieldInitialization(lhs, field, field->getName());
  }
}

static void buildMemberInitializer(CIRGenFunction &cgf,
                                   const CXXRecordDecl *classDecl,
                                   CXXCtorInitializer *memberInit,
                                   const CXXConstructorDecl *constructor,
                                   FunctionArgList &args) {
  // TODO: ApplyDebugLocation
  assert(memberInit->isAnyMemberInitializer() &&
         "Mush have member initializer!");
  assert(memberInit->getInit() && "Must have initializer!");

  // non-static data member initializers
  FieldDecl *field = memberInit->getAnyMember();
  QualType fieldType = field->getType();

  auto thisPtr = cgf.LoadCXXThis();
  QualType recordTy = cgf.getContext().getTypeDeclType(classDecl);
  LValue lhs;

  // If a base constructor is being emitted, create an LValue that has the
  // non-virtual alignment.
  if (cgf.CurGD.getCtorType() == Ctor_Base)
    lhs = cgf.MakeNaturalAlignPointeeAddrLValue(thisPtr, recordTy);
  else
    lhs = cgf.MakeNaturalAlignAddrLValue(thisPtr, recordTy);

  buildLValueForAnyFieldInitialization(cgf, memberInit, lhs);

  // Special case: If we are in a copy or move constructor, and we are copying
  // an array off PODs or classes with tirival copy constructors, ignore the AST
  // and perform the copy we know is equivalent.
  // FIXME: This is hacky at best... if we had a bit more explicit information
  // in the AST, we could generalize it more easily.
  const ConstantArrayType *array =
      cgf.getContext().getAsConstantArrayType(fieldType);
  if (array && constructor->isDefaulted() &&
      constructor->isCopyOrMoveConstructor()) {
    llvm_unreachable("NYI");
  }

  cgf.buildInitializerForField(field, lhs, memberInit->getInit());
}

class ConstructorMemcpyizer : public FieldMemcpyizer {
private:
  /// Get source argument for copy constructor. Returns null if not a copy
  /// constructor.
  static const VarDecl *getTrivialCopySource(CIRGenFunction &cgf,
                                             const CXXConstructorDecl *cd,
                                             FunctionArgList &args) {
    if (cd->isCopyOrMoveConstructor() && cd->isDefaulted())
      return args[cgf.cgm.getCXXABI().getSrcArgforCopyCtor(cd, args)];

    return nullptr;
  }

  // Returns true if a CXXCtorInitializer represents a member initialization
  // that can be rolled into a memcpy.
  bool isMemberInitMemcpyable(CXXCtorInitializer *memberInit) const {
    if (!memcpyableCtor)
      return false;

    assert(!MissingFeatures::fieldMemcpyizerBuildMemcpy());
    return false;
  }

public:
  ConstructorMemcpyizer(CIRGenFunction &cgf, const CXXConstructorDecl *cd,
                        FunctionArgList &args)
      : FieldMemcpyizer(cgf, cd->getParent(),
                        getTrivialCopySource(cgf, cd, args)),
        constructorDecl(cd),
        memcpyableCtor(cd->isDefaulted() && cd->isCopyOrMoveConstructor() &&
                       cgf.getLangOpts().getGC() == LangOptions::NonGC),
        args(args) {}

  void addMemberInitializer(CXXCtorInitializer *memberInit) {
    if (isMemberInitMemcpyable(memberInit)) {
      AggregatedInits.push_back(memberInit);
      addMemcpyableField(memberInit->getMember());
    } else {
      buildAggregatedInits();
      buildMemberInitializer(cgf, constructorDecl->getParent(), memberInit,
                             constructorDecl, args);
    }
  }

  void buildAggregatedInits() {
    if (AggregatedInits.size() <= 1) {
      // This memcpy is too small to be worthwhile. Fall back on default
      // codegen.
      if (!AggregatedInits.empty()) {
        llvm_unreachable("NYI");
      }
      reset();
      return;
    }

    pushEHDestructors();
    buildMemcpy();
    AggregatedInits.clear();
  }

  void pushEHDestructors() {
    Address thisPtr = cgf.LoadCXXThisAddress();
    QualType recordTy = cgf.getContext().getTypeDeclType(classDecl);
    LValue lhs = cgf.makeAddrLValue(thisPtr, recordTy);
    (void)lhs;

    for (auto MemberInit : AggregatedInits) {
      QualType fieldType = MemberInit->getAnyMember()->getType();
      QualType::DestructionKind dtorKind = fieldType.isDestructedType();
      if (!cgf.needsEHCleanup(dtorKind))
        continue;
      LValue fieldLhs = lhs;
      buildLValueForAnyFieldInitialization(cgf, MemberInit, fieldLhs);
      cgf.pushEHDestroy(dtorKind, fieldLhs.getAddress(), fieldType);
    }
  }

  void finish() { buildAggregatedInits(); }

private:
  const CXXConstructorDecl *constructorDecl;
  bool memcpyableCtor;
  FunctionArgList &args;
  SmallVector<CXXCtorInitializer *, 16> AggregatedInits;
};

class AssignmentMemcpyizer : public FieldMemcpyizer {
private:
  // Returns the memcpyable field copied by the given statement, if one
  // exists. Otherwise returns null.
  FieldDecl *getMemcpyableField(Stmt *S) {
    if (!assignmentsMemcpyable)
      return nullptr;
    if (BinaryOperator *bo = dyn_cast<BinaryOperator>(S)) {
      // Recognise trivial assignments.
      if (bo->getOpcode() != BO_Assign)
        return nullptr;
      MemberExpr *me = dyn_cast<MemberExpr>(bo->getLHS());
      if (!me)
        return nullptr;
      FieldDecl *field = dyn_cast<FieldDecl>(me->getMemberDecl());
      if (!field || !isMemcpyableField(field))
        return nullptr;
      Stmt *rhs = bo->getRHS();
      if (ImplicitCastExpr *ec = dyn_cast<ImplicitCastExpr>(rhs))
        rhs = ec->getSubExpr();
      if (!rhs)
        return nullptr;
      if (MemberExpr *mE2 = dyn_cast<MemberExpr>(rhs)) {
        if (mE2->getMemberDecl() == field)
          return field;
      }
      return nullptr;
    }
    if (CXXMemberCallExpr *MCE = dyn_cast<CXXMemberCallExpr>(S)) {
      CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MCE->getCalleeDecl());
      if (!(MD && isMemcpyEquivalentSpecialMember(MD)))
        return nullptr;
      MemberExpr *IOA = dyn_cast<MemberExpr>(MCE->getImplicitObjectArgument());
      if (!IOA)
        return nullptr;
      FieldDecl *Field = dyn_cast<FieldDecl>(IOA->getMemberDecl());
      if (!Field || !isMemcpyableField(Field))
        return nullptr;
      MemberExpr *Arg0 = dyn_cast<MemberExpr>(MCE->getArg(0));
      if (!Arg0 || Field != dyn_cast<FieldDecl>(Arg0->getMemberDecl()))
        return nullptr;
      return Field;
    } else if (CallExpr *CE = dyn_cast<CallExpr>(S)) {
      FunctionDecl *FD = dyn_cast<FunctionDecl>(CE->getCalleeDecl());
      if (!FD || FD->getBuiltinID() != Builtin::BI__builtin_memcpy)
        return nullptr;
      Expr *DstPtr = CE->getArg(0);
      if (ImplicitCastExpr *DC = dyn_cast<ImplicitCastExpr>(DstPtr))
        DstPtr = DC->getSubExpr();
      UnaryOperator *DUO = dyn_cast<UnaryOperator>(DstPtr);
      if (!DUO || DUO->getOpcode() != UO_AddrOf)
        return nullptr;
      MemberExpr *ME = dyn_cast<MemberExpr>(DUO->getSubExpr());
      if (!ME)
        return nullptr;
      FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
      if (!Field || !isMemcpyableField(Field))
        return nullptr;
      Expr *SrcPtr = CE->getArg(1);
      if (ImplicitCastExpr *SC = dyn_cast<ImplicitCastExpr>(SrcPtr))
        SrcPtr = SC->getSubExpr();
      UnaryOperator *SUO = dyn_cast<UnaryOperator>(SrcPtr);
      if (!SUO || SUO->getOpcode() != UO_AddrOf)
        return nullptr;
      MemberExpr *ME2 = dyn_cast<MemberExpr>(SUO->getSubExpr());
      if (!ME2 || Field != dyn_cast<FieldDecl>(ME2->getMemberDecl()))
        return nullptr;
      return Field;
    }

    return nullptr;
  }

  bool assignmentsMemcpyable;
  SmallVector<Stmt *, 16> aggregatedStmts;

public:
  AssignmentMemcpyizer(CIRGenFunction &cgf, const CXXMethodDecl *ad,
                       FunctionArgList &args)
      : FieldMemcpyizer(cgf, ad->getParent(), args[args.size() - 1]),
        assignmentsMemcpyable(cgf.getLangOpts().getGC() == LangOptions::NonGC) {
    assert(args.size() == 2);
  }

  void emitAssignment(Stmt *s) {
    FieldDecl *f = getMemcpyableField(s);
    if (f) {
      addMemcpyableField(f);
      aggregatedStmts.push_back(s);
    } else {
      emitAggregatedStmts();
      if (cgf.buildStmt(s, /*useCurrentScope=*/true).failed())
        llvm_unreachable("Should not get here!");
    }
  }

  void emitAggregatedStmts() {
    if (aggregatedStmts.size() <= 1) {
      if (!aggregatedStmts.empty()) {
        CopyingValueRepresentation cvr(cgf);
        if (cgf.buildStmt(aggregatedStmts[0], /*useCurrentScope=*/true)
                .failed())
          llvm_unreachable("Should not get here!");
      }
      reset();
    }

    buildMemcpy();
    aggregatedStmts.clear();
  }

  void finish() { emitAggregatedStmts(); }
};
} // namespace

static bool isInitializerOfDynamicClass(const CXXCtorInitializer *baseInit) {
  const Type *baseType = baseInit->getBaseClass();
  const auto *baseClassDecl =
      cast<CXXRecordDecl>(baseType->castAs<RecordType>()->getDecl());
  return baseClassDecl->isDynamicClass();
}

namespace {
/// Call the destructor for a direct base class.
struct CallBaseDtor final : EHScopeStack::Cleanup {
  const CXXRecordDecl *baseClass;
  bool baseIsVirtual;
  CallBaseDtor(const CXXRecordDecl *base, bool baseIsVirtual)
      : baseClass(base), baseIsVirtual(baseIsVirtual) {}

  void Emit(CIRGenFunction &cgf, Flags flags) override {
    const CXXRecordDecl *derivedClass =
        cast<CXXMethodDecl>(cgf.CurCodeDecl)->getParent();

    const CXXDestructorDecl *d = baseClass->getDestructor();
    // We are already inside a destructor, so presumably the object being
    // destroyed should have the expected type.
    QualType thisTy = d->getFunctionObjectParameterType();
    assert(cgf.currSrcLoc && "expected source location");
    Address addr = cgf.getAddressOfDirectBaseInCompleteClass(
        *cgf.currSrcLoc, cgf.LoadCXXThisAddress(), derivedClass, baseClass,
        baseIsVirtual);
    cgf.buildCXXDestructorCall(d, Dtor_Base, baseIsVirtual,
                               /*Delegating=*/false, addr, thisTy);
  }
};

/// A visitor which checks whether an initializer uses 'this' in a
/// way which requires the vtable to be properly set.
struct DynamicThisUseChecker
    : ConstEvaluatedExprVisitor<DynamicThisUseChecker> {
  using super = ConstEvaluatedExprVisitor<DynamicThisUseChecker>;

  bool UsesThis = false;

  DynamicThisUseChecker(const ASTContext &c) : super(c) {}

  // Black-list all explicit and implicit references to 'this'.
  //
  // Do we need to worry about external references to 'this' derived
  // from arbitrary code?  If so, then anything which runs arbitrary
  // external code might potentially access the vtable.
  void VisitCXXThisExpr(const CXXThisExpr *e) { UsesThis = true; }
};
} // end anonymous namespace

static bool baseInitializerUsesThis(ASTContext &c, const Expr *init) {
  DynamicThisUseChecker checker(c);
  checker.Visit(init);
  return checker.UsesThis;
}

/// Gets the address of a direct base class within a complete object.
/// This should only be used for (1) non-virtual bases or (2) virtual bases
/// when the type is known to be complete (e.g. in complete destructors).
///
/// The object pointed to by 'This' is assumed to be non-null.
Address CIRGenFunction::getAddressOfDirectBaseInCompleteClass(
    mlir::Location loc, Address This, const CXXRecordDecl *derived,
    const CXXRecordDecl *base, bool baseIsVirtual) {
  // 'this' must be a pointer (in some address space) to Derived.
  assert(This.getElementType() == ConvertType(derived));

  // Compute the offset of the virtual base.
  CharUnits offset;
  const ASTRecordLayout &layout = getContext().getASTRecordLayout(derived);
  if (baseIsVirtual)
    offset = layout.getVBaseClassOffset(base);
  else
    offset = layout.getBaseClassOffset(base);

  return builder.createBaseClassAddr(loc, This, ConvertType(base),
                                     offset.getQuantity(),
                                     /*assumeNotNull=*/true);
}

static void buildBaseInitializer(mlir::Location loc, CIRGenFunction &cgf,
                                 const CXXRecordDecl *classDecl,
                                 CXXCtorInitializer *baseInit) {
  assert(baseInit->isBaseInitializer() && "Must have base initializer!");

  Address thisPtr = cgf.LoadCXXThisAddress();

  const Type *baseType = baseInit->getBaseClass();
  const auto *baseClassDecl =
      cast<CXXRecordDecl>(baseType->castAs<RecordType>()->getDecl());

  bool isBaseVirtual = baseInit->isBaseVirtual();

  // If the initializer for the base (other than the constructor
  // itself) accesses 'this' in any way, we need to initialize the
  // vtables.
  if (baseInitializerUsesThis(cgf.getContext(), baseInit->getInit()))
    cgf.initializeVTablePointers(loc, classDecl);

  // We can pretend to be a complete class because it only matters for
  // virtual bases, and we only do virtual bases for complete ctors.
  Address v = cgf.getAddressOfDirectBaseInCompleteClass(
      loc, thisPtr, classDecl, baseClassDecl, isBaseVirtual);
  AggValueSlot aggSlot = AggValueSlot::forAddr(
      v, Qualifiers(), AggValueSlot::IsDestructed,
      AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
      cgf.getOverlapForBaseInit(classDecl, baseClassDecl, isBaseVirtual));

  cgf.buildAggExpr(baseInit->getInit(), aggSlot);

  if (cgf.cgm.getLangOpts().Exceptions &&
      !baseClassDecl->hasTrivialDestructor())
    cgf.EHStack.pushCleanup<CallBaseDtor>(EHCleanup, baseClassDecl,
                                          isBaseVirtual);
}

/// This routine generates necessary code to initialize base classes and
/// non-static data members belonging to this constructor.
void CIRGenFunction::buildCtorPrologue(const CXXConstructorDecl *cd,
                                       CXXCtorType ctorType,
                                       FunctionArgList &args) {
  if (cd->isDelegatingConstructor())
    return buildDelegatingCXXConstructorCall(cd, args);

  const CXXRecordDecl *classDecl = cd->getParent();

  CXXConstructorDecl::init_const_iterator b = cd->init_begin(),
                                          e = cd->init_end();

  // Virtual base initializers first, if any. They aren't needed if:
  // - This is a base ctor variant
  // - There are no vbases
  // - The class is abstract, so a complete object of it cannot be constructed
  //
  // The check for an abstract class is necessary because sema may not have
  // marked virtual base destructors referenced.
  bool constructVBases = ctorType != Ctor_Base &&
                         classDecl->getNumVBases() != 0 &&
                         !classDecl->isAbstract();

  // In the Microsoft C++ ABI, there are no constructor variants. Instead, the
  // constructor of a class with virtual bases takes an additional parameter to
  // conditionally construct the virtual bases. Emit that check here.
  mlir::Block *baseCtorContinueBb = nullptr;
  if (constructVBases &&
      !cgm.getTarget().getCXXABI().hasConstructorVariants()) {
    llvm_unreachable("NYI");
  }

  auto const oldThis = CXXThisValue;
  for (; b != e && (*b)->isBaseInitializer() && (*b)->isBaseVirtual(); b++) {
    if (!constructVBases)
      continue;
    if (cgm.getCodeGenOpts().StrictVTablePointers &&
        cgm.getCodeGenOpts().OptimizationLevel > 0 &&
        isInitializerOfDynamicClass(*b))
      llvm_unreachable("NYI");
    buildBaseInitializer(getLoc(cd->getBeginLoc()), *this, classDecl, *b);
  }

  if (baseCtorContinueBb) {
    llvm_unreachable("NYI");
  }

  // Then, non-virtual base initializers.
  for (; b != e && (*b)->isBaseInitializer(); b++) {
    assert(!(*b)->isBaseVirtual());

    if (cgm.getCodeGenOpts().StrictVTablePointers &&
        cgm.getCodeGenOpts().OptimizationLevel > 0 &&
        isInitializerOfDynamicClass(*b))
      llvm_unreachable("NYI");
    buildBaseInitializer(getLoc(cd->getBeginLoc()), *this, classDecl, *b);
  }

  CXXThisValue = oldThis;

  initializeVTablePointers(getLoc(cd->getBeginLoc()), classDecl);

  // And finally, initialize class members.
  FieldConstructionScope fcs(*this, LoadCXXThisAddress());
  ConstructorMemcpyizer cm(*this, cd, args);
  for (; b != e; b++) {
    CXXCtorInitializer *member = (*b);
    assert(!member->isBaseInitializer());
    assert(member->isAnyMemberInitializer() &&
           "Delegating initializer on non-delegating constructor");
    cm.addMemberInitializer(member);
  }
  cm.finish();
}

static Address applyNonVirtualAndVirtualOffset(
    mlir::Location loc, CIRGenFunction &cgf, Address addr,
    CharUnits nonVirtualOffset, mlir::Value virtualOffset,
    const CXXRecordDecl *derivedClass, const CXXRecordDecl *nearestVBase) {
  // Assert that we have something to do.
  assert(!nonVirtualOffset.isZero() || virtualOffset != nullptr);

  // Compute the offset from the static and dynamic components.
  mlir::Value baseOffset;
  if (!nonVirtualOffset.isZero()) {
    mlir::Type offsetType =
        (cgf.cgm.getTarget().getCXXABI().isItaniumFamily() &&
         cgf.cgm.getItaniumVTableContext().isRelativeLayout())
            ? cgf.SInt32Ty
            : cgf.PtrDiffTy;
    baseOffset = cgf.getBuilder().getConstInt(loc, offsetType,
                                              nonVirtualOffset.getQuantity());
    if (virtualOffset) {
      baseOffset = cgf.getBuilder().createBinop(
          virtualOffset, mlir::cir::BinOpKind::Add, baseOffset);
    }
  } else {
    baseOffset = virtualOffset;
  }

  // Apply the base offset.  cir.ptr_stride adjusts by a number of elements,
  // not bytes.  So the pointer must be cast to a byte pointer and back.

  mlir::Value ptr = addr.getPointer();
  mlir::Type charPtrType = cgf.cgm.UInt8PtrTy;
  mlir::Value charPtr = cgf.getBuilder().createCast(
      mlir::cir::CastKind::bitcast, ptr, charPtrType);
  mlir::Value adjusted = cgf.getBuilder().create<mlir::cir::PtrStrideOp>(
      loc, charPtrType, charPtr, baseOffset);
  ptr = cgf.getBuilder().createCast(mlir::cir::CastKind::bitcast, adjusted,
                                    ptr.getType());

  // If we have a virtual component, the alignment of the result will
  // be relative only to the known alignment of that vbase.
  CharUnits alignment;
  if (virtualOffset) {
    assert(nearestVBase && "virtual offset without vbase?");
    llvm_unreachable("NYI");
    // alignment = CGF.CGM.getVBaseAlignment(addr.getAlignment(),
    //                                       derivedClass, nearestVBase);
  } else {
    alignment = addr.getAlignment();
  }
  alignment = alignment.alignmentAtOffset(nonVirtualOffset);

  return Address(ptr, alignment);
}

void CIRGenFunction::initializeVTablePointer(mlir::Location loc,
                                             const VPtr &vptr) {
  // Compute the address point.
  auto vTableAddressPoint = cgm.getCXXABI().getVTableAddressPointInStructor(
      *this, vptr.VTableClass, vptr.Base, vptr.NearestVBase);

  if (!vTableAddressPoint)
    return;

  // Compute where to store the address point.
  mlir::Value virtualOffset{};
  CharUnits nonVirtualOffset = CharUnits::Zero();

  if (cgm.getCXXABI().isVirtualOffsetNeededForVTableField(*this, vptr)) {
    llvm_unreachable("NYI");
  } else {
    // We can just use the base offset in the complete class.
    nonVirtualOffset = vptr.Base.getBaseOffset();
  }

  // Apply the offsets.
  Address vTableField = LoadCXXThisAddress();
  if (!nonVirtualOffset.isZero() || virtualOffset) {
    vTableField = applyNonVirtualAndVirtualOffset(
        loc, *this, vTableField, nonVirtualOffset, virtualOffset,
        vptr.VTableClass, vptr.NearestVBase);
  }

  // Finally, store the address point. Use the same CIR types as the field.
  //
  // vtable field is derived from `this` pointer, therefore they should be in
  // the same addr space.
  assert(!MissingFeatures::addressSpace());
  vTableField = builder.createElementBitCast(loc, vTableField,
                                             vTableAddressPoint.getType());
  builder.createStore(loc, vTableAddressPoint, vTableField);
  assert(!MissingFeatures::tbaa());
}

void CIRGenFunction::initializeVTablePointers(mlir::Location loc,
                                              const CXXRecordDecl *rd) {
  // Ignore classes without a vtable.
  if (!rd->isDynamicClass())
    return;

  // Initialize the vtable pointers for this class and all of its bases.
  if (cgm.getCXXABI().doStructorsInitializeVPtrs(rd))
    for (const auto &vptr : getVTablePointers(rd))
      initializeVTablePointer(loc, vptr);

  if (rd->getNumVBases())
    cgm.getCXXABI().initializeHiddenVirtualInheritanceMembers(*this, rd);
}

CIRGenFunction::VPtrsVector
CIRGenFunction::getVTablePointers(const CXXRecordDecl *vTableClass) {
  CIRGenFunction::VPtrsVector vPtrsResult;
  VisitedVirtualBasesSetTy vBases;
  getVTablePointers(BaseSubobject(vTableClass, CharUnits::Zero()),
                    /*NearestVBase=*/nullptr,
                    /*OffsetFromNearestVBase=*/CharUnits::Zero(),
                    /*BaseIsNonVirtualPrimaryBase=*/false, vTableClass, vBases,
                    vPtrsResult);
  return vPtrsResult;
}

void CIRGenFunction::getVTablePointers(BaseSubobject base,
                                       const CXXRecordDecl *nearestVBase,
                                       CharUnits offsetFromNearestVBase,
                                       bool baseIsNonVirtualPrimaryBase,
                                       const CXXRecordDecl *vTableClass,
                                       VisitedVirtualBasesSetTy &vBases,
                                       VPtrsVector &vptrs) {
  // If this base is a non-virtual primary base the address point has already
  // been set.
  if (!baseIsNonVirtualPrimaryBase) {
    // Initialize the vtable pointer for this base.
    VPtr vptr = {base, nearestVBase, offsetFromNearestVBase, vTableClass};
    vptrs.push_back(vptr);
  }

  const CXXRecordDecl *rd = base.getBase();

  // Traverse bases.
  for (const auto &i : rd->bases()) {
    auto *baseDecl =
        cast<CXXRecordDecl>(i.getType()->castAs<RecordType>()->getDecl());

    // Ignore classes without a vtable.
    if (!baseDecl->isDynamicClass())
      continue;

    CharUnits baseOffset;
    CharUnits baseOffsetFromNearestVBase;
    bool baseDeclIsNonVirtualPrimaryBase;

    if (i.isVirtual()) {
      llvm_unreachable("NYI");
    } else {
      const ASTRecordLayout &layout = getContext().getASTRecordLayout(rd);

      baseOffset = base.getBaseOffset() + layout.getBaseClassOffset(baseDecl);
      baseOffsetFromNearestVBase =
          offsetFromNearestVBase + layout.getBaseClassOffset(baseDecl);
      baseDeclIsNonVirtualPrimaryBase = layout.getPrimaryBase() == baseDecl;
    }

    getVTablePointers(
        BaseSubobject(baseDecl, baseOffset),
        i.isVirtual() ? baseDecl : nearestVBase, baseOffsetFromNearestVBase,
        baseDeclIsNonVirtualPrimaryBase, vTableClass, vBases, vptrs);
  }
}

Address CIRGenFunction::LoadCXXThisAddress() {
  assert(CurFuncDecl && "loading 'this' without a func declaration?");
  assert(isa<CXXMethodDecl>(CurFuncDecl));

  // Lazily compute CXXThisAlignment.
  if (CXXThisAlignment.isZero()) {
    // Just use the best known alignment for the parent.
    // TODO: if we're currently emitting a complete-object ctor/dtor, we can
    // always use the complete-object alignment.
    const auto *rd = cast<CXXMethodDecl>(CurFuncDecl)->getParent();
    CXXThisAlignment = cgm.getClassPointerAlignment(rd);
  }

  return Address(LoadCXXThis(), CXXThisAlignment);
}

void CIRGenFunction::buildInitializerForField(FieldDecl *field, LValue lhs,
                                              Expr *init) {
  QualType fieldType = field->getType();
  switch (getEvaluationKind(fieldType)) {
  case TEK_Scalar:
    if (lhs.isSimple()) {
      buildExprAsInit(init, field, lhs, false);
    } else {
      llvm_unreachable("NYI");
    }
    break;
  case TEK_Complex:
    llvm_unreachable("NYI");
    break;
  case TEK_Aggregate: {
    AggValueSlot slot = AggValueSlot::forLValue(
        lhs, AggValueSlot::IsDestructed, AggValueSlot::DoesNotNeedGCBarriers,
        AggValueSlot::IsNotAliased, getOverlapForFieldInit(field),
        AggValueSlot::IsNotZeroed,
        // Checks are made by the code that calls constructor.
        AggValueSlot::IsSanitizerChecked);
    buildAggExpr(init, slot);
    break;
  }
  }

  // Ensure that we destroy this object if an exception is thrown later in the
  // constructor.
  QualType::DestructionKind dtorKind = fieldType.isDestructedType();
  (void)dtorKind;
  if (MissingFeatures::cleanups())
    llvm_unreachable("NYI");
}

void CIRGenFunction::buildDelegateCXXConstructorCall(
    const CXXConstructorDecl *ctor, CXXCtorType ctorType,
    const FunctionArgList &args, SourceLocation loc) {
  CallArgList delegateArgs;

  FunctionArgList::const_iterator i = args.begin(), e = args.end();
  assert(i != e && "no parameters to constructor");

  // this
  Address This = LoadCXXThisAddress();
  delegateArgs.add(RValue::get(This.getPointer()), (*i)->getType());
  ++i;

  // FIXME: The location of the VTT parameter in the parameter list is specific
  // to the Itanium ABI and shouldn't be hardcoded here.
  if (cgm.getCXXABI().NeedsVTTParameter(CurGD)) {
    llvm_unreachable("NYI");
  }

  // Explicit arguments.
  for (; i != e; ++i) {
    const VarDecl *param = *i;
    // FIXME: per-argument source location
    buildDelegateCallArg(delegateArgs, param, loc);
  }

  buildCXXConstructorCall(ctor, ctorType, /*ForVirtualBase=*/false,
                          /*Delegating=*/true, This, delegateArgs,
                          AggValueSlot::MayOverlap, loc,
                          /*NewPointerIsChecked=*/true);
}

void CIRGenFunction::buildImplicitAssignmentOperatorBody(
    FunctionArgList &args) {
  const CXXMethodDecl *assignOp = cast<CXXMethodDecl>(CurGD.getDecl());
  const Stmt *rootS = assignOp->getBody();
  assert(isa<CompoundStmt>(rootS) &&
         "Body of an implicit assignment operator should be compound stmt.");
  const CompoundStmt *rootCs = cast<CompoundStmt>(rootS);

  // LexicalScope Scope(*this, RootCS->getSourceRange());
  // FIXME(cir): add all of the below under a new scope.

  assert(!MissingFeatures::incrementProfileCounter());
  AssignmentMemcpyizer am(*this, assignOp, args);
  for (auto *i : rootCs->body())
    am.emitAssignment(i);
  am.finish();
}

void CIRGenFunction::buildForwardingCallToLambda(
    const CXXMethodDecl *callOperator, CallArgList &callArgs) {
  // Get the address of the call operator.
  const auto &calleeFnInfo =
      cgm.getTypes().arrangeCXXMethodDeclaration(callOperator);
  auto calleePtr = cgm.GetAddrOfFunction(
      GlobalDecl(callOperator), cgm.getTypes().GetFunctionType(calleeFnInfo));

  // Prepare the return slot.
  const FunctionProtoType *fpt =
      callOperator->getType()->castAs<FunctionProtoType>();
  QualType resultType = fpt->getReturnType();
  ReturnValueSlot returnSlot;
  if (!resultType->isVoidType() &&
      calleeFnInfo.getReturnInfo().getKind() == ABIArgInfo::Indirect &&
      !hasScalarEvaluationKind(calleeFnInfo.getReturnType())) {
    llvm_unreachable("NYI");
  }

  // We don't need to separately arrange the call arguments because
  // the call can't be variadic anyway --- it's impossible to forward
  // variadic arguments.

  // Now emit our call.
  auto callee = CIRGenCallee::forDirect(calleePtr, GlobalDecl(callOperator));
  RValue rv = buildCall(calleeFnInfo, callee, returnSlot, callArgs);

  // If necessary, copy the returned value into the slot.
  if (!resultType->isVoidType() && returnSlot.isNull()) {
    if (getLangOpts().ObjCAutoRefCount && resultType->isObjCRetainableType())
      llvm_unreachable("NYI");
    buildReturnOfRValue(*currSrcLoc, rv, resultType);
  } else {
    llvm_unreachable("NYI");
  }
}

void CIRGenFunction::buildLambdaDelegatingInvokeBody(const CXXMethodDecl *md) {
  const CXXRecordDecl *lambda = md->getParent();

  // Start building arguments for forwarding call
  CallArgList callArgs;

  QualType lambdaType = getContext().getRecordType(lambda);
  QualType thisType = getContext().getPointerType(lambdaType);
  Address thisPtr =
      CreateMemTemp(lambdaType, getLoc(md->getSourceRange()), "unused.capture");
  callArgs.add(RValue::get(thisPtr.getPointer()), thisType);

  // Add the rest of the parameters.
  for (auto *param : md->parameters())
    buildDelegateCallArg(callArgs, param, param->getBeginLoc());

  const CXXMethodDecl *callOp = lambda->getLambdaCallOperator();
  // For a generic lambda, find the corresponding call operator specialization
  // to which the call to the static-invoker shall be forwarded.
  if (lambda->isGenericLambda()) {
    assert(md->isFunctionTemplateSpecialization());
    const TemplateArgumentList *tal = md->getTemplateSpecializationArgs();
    FunctionTemplateDecl *callOpTemplate =
        callOp->getDescribedFunctionTemplate();
    void *insertPos = nullptr;
    FunctionDecl *correspondingCallOpSpecialization =
        callOpTemplate->findSpecialization(tal->asArray(), insertPos);
    assert(correspondingCallOpSpecialization);
    callOp = cast<CXXMethodDecl>(correspondingCallOpSpecialization);
  }
  buildForwardingCallToLambda(callOp, callArgs);
}

void CIRGenFunction::buildLambdaStaticInvokeBody(const CXXMethodDecl *md) {
  if (md->isVariadic()) {
    // Codgen for LLVM doesn't emit code for this as well, it says:
    // FIXME: Making this work correctly is nasty because it requires either
    // cloning the body of the call operator or making the call operator
    // forward.
    llvm_unreachable("NYI");
  }

  buildLambdaDelegatingInvokeBody(md);
}

void CIRGenFunction::destroyCXXObject(CIRGenFunction &cgf, Address addr,
                                      QualType type) {
  const RecordType *rtype = type->castAs<RecordType>();
  const CXXRecordDecl *record = cast<CXXRecordDecl>(rtype->getDecl());
  const CXXDestructorDecl *dtor = record->getDestructor();
  // TODO(cir): Unlike traditional codegen, CIRGen should actually emit trivial
  // dtors which shall be removed on later CIR passes. However, only remove this
  // assertion once we get a testcase to exercise this path.
  assert(!dtor->isTrivial());
  cgf.buildCXXDestructorCall(dtor, Dtor_Complete, /*for vbase*/ false,
                             /*Delegating=*/false, addr, type);
}

static bool fieldHasTrivialDestructorBody(ASTContext &context,
                                          const FieldDecl *field);

// FIXME(cir): this should be shared with traditional codegen.
static bool
hasTrivialDestructorBody(ASTContext &context,
                         const CXXRecordDecl *baseClassDecl,
                         const CXXRecordDecl *mostDerivedClassDecl) {
  // If the destructor is trivial we don't have to check anything else.
  if (baseClassDecl->hasTrivialDestructor())
    return true;

  if (!baseClassDecl->getDestructor()->hasTrivialBody())
    return false;

  // Check fields.
  for (const auto *field : baseClassDecl->fields())
    if (!fieldHasTrivialDestructorBody(context, field))
      return false;

  // Check non-virtual bases.
  for (const auto &i : baseClassDecl->bases()) {
    if (i.isVirtual())
      continue;

    const CXXRecordDecl *nonVirtualBase =
        cast<CXXRecordDecl>(i.getType()->castAs<RecordType>()->getDecl());
    if (!hasTrivialDestructorBody(context, nonVirtualBase,
                                  mostDerivedClassDecl))
      return false;
  }

  if (baseClassDecl == mostDerivedClassDecl) {
    // Check virtual bases.
    for (const auto &i : baseClassDecl->vbases()) {
      const CXXRecordDecl *virtualBase =
          cast<CXXRecordDecl>(i.getType()->castAs<RecordType>()->getDecl());
      if (!hasTrivialDestructorBody(context, virtualBase, mostDerivedClassDecl))
        return false;
    }
  }

  return true;
}

// FIXME(cir): this should be shared with traditional codegen.
static bool fieldHasTrivialDestructorBody(ASTContext &context,
                                          const FieldDecl *field) {
  QualType fieldBaseElementType = context.getBaseElementType(field->getType());

  const RecordType *rt = fieldBaseElementType->getAs<RecordType>();
  if (!rt)
    return true;

  CXXRecordDecl *fieldClassDecl = cast<CXXRecordDecl>(rt->getDecl());

  // The destructor for an implicit anonymous union member is never invoked.
  if (fieldClassDecl->isUnion() && fieldClassDecl->isAnonymousStructOrUnion())
    return false;

  return hasTrivialDestructorBody(context, fieldClassDecl, fieldClassDecl);
}

/// Check whether we need to initialize any vtable pointers before calling this
/// destructor.
/// FIXME(cir): this should be shared with traditional codegen.
static bool canSkipVTablePointerInitialization(CIRGenFunction &cgf,
                                               const CXXDestructorDecl *dtor) {
  const CXXRecordDecl *classDecl = dtor->getParent();
  if (!classDecl->isDynamicClass())
    return true;

  // For a final class, the vtable pointer is known to already point to the
  // class's vtable.
  if (classDecl->isEffectivelyFinal())
    return true;

  if (!dtor->hasTrivialBody())
    return false;

  // Check the fields.
  for (const auto *field : classDecl->fields())
    if (!fieldHasTrivialDestructorBody(cgf.getContext(), field))
      return false;

  return true;
}

/// Emits the body of the current destructor.
void CIRGenFunction::buildDestructorBody(FunctionArgList &args) {
  const CXXDestructorDecl *dtor = cast<CXXDestructorDecl>(CurGD.getDecl());
  CXXDtorType dtorType = CurGD.getDtorType();

  // For an abstract class, non-base destructors are never used (and can't
  // be emitted in general, because vbase dtors may not have been validated
  // by Sema), but the Itanium ABI doesn't make them optional and Clang may
  // in fact emit references to them from other compilations, so emit them
  // as functions containing a trap instruction.
  if (dtorType != Dtor_Base && dtor->getParent()->isAbstract()) {
    SourceLocation loc =
        dtor->hasBody() ? dtor->getBody()->getBeginLoc() : dtor->getLocation();
    builder.create<mlir::cir::TrapOp>(getLoc(loc));
    // The corresponding clang/CodeGen logic clears the insertion point here,
    // but MLIR's builder requires a valid insertion point, so we create a dummy
    // block (since the trap is a block terminator).
    builder.createBlock(builder.getBlock()->getParent());
    return;
  }

  Stmt *Body = dtor->getBody();
  if (Body)
    assert(!MissingFeatures::incrementProfileCounter());

  // The call to operator delete in a deleting destructor happens
  // outside of the function-try-block, which means it's always
  // possible to delegate the destructor body to the complete
  // destructor.  Do so.
  if (dtorType == Dtor_Deleting) {
    RunCleanupsScope dtorEpilogue(*this);
    EnterDtorCleanups(dtor, Dtor_Deleting);
    if (HaveInsertPoint()) {
      QualType thisTy = dtor->getFunctionObjectParameterType();
      buildCXXDestructorCall(dtor, Dtor_Complete, /*ForVirtualBase=*/false,
                             /*Delegating=*/false, LoadCXXThisAddress(),
                             thisTy);
    }
    return;
  }

  // If the body is a function-try-block, enter the try before
  // anything else.
  bool isTryBody = (isa_and_nonnull<CXXTryStmt>(Body));
  if (isTryBody) {
    llvm_unreachable("NYI");
    // EnterCXXTryStmt(*cast<CXXTryStmt>(Body), true);
  }
  if (MissingFeatures::emitAsanPrologueOrEpilogue())
    llvm_unreachable("NYI");

  // Enter the epilogue cleanups.
  RunCleanupsScope dtorEpilogue(*this);

  // If this is the complete variant, just invoke the base variant;
  // the epilogue will destruct the virtual bases.  But we can't do
  // this optimization if the body is a function-try-block, because
  // we'd introduce *two* handler blocks.  In the Microsoft ABI, we
  // always delegate because we might not have a definition in this TU.
  switch (dtorType) {
  case Dtor_Comdat:
    llvm_unreachable("not expecting a COMDAT");
  case Dtor_Deleting:
    llvm_unreachable("already handled deleting case");

  case Dtor_Complete:
    assert((Body || getTarget().getCXXABI().isMicrosoft()) &&
           "can't emit a dtor without a body for non-Microsoft ABIs");

    // Enter the cleanup scopes for virtual bases.
    EnterDtorCleanups(dtor, Dtor_Complete);

    if (!isTryBody) {
      QualType thisTy = dtor->getFunctionObjectParameterType();
      buildCXXDestructorCall(dtor, Dtor_Base, /*ForVirtualBase=*/false,
                             /*Delegating=*/false, LoadCXXThisAddress(),
                             thisTy);
      break;
    }

    // Fallthrough: act like we're in the base variant.
    [[fallthrough]];

  case Dtor_Base:
    assert(Body);

    // Enter the cleanup scopes for fields and non-virtual bases.
    EnterDtorCleanups(dtor, Dtor_Base);

    // Initialize the vtable pointers before entering the body.
    if (!canSkipVTablePointerInitialization(*this, dtor)) {
      // Insert the llvm.launder.invariant.group intrinsic before initializing
      // the vptrs to cancel any previous assumptions we might have made.
      if (cgm.getCodeGenOpts().StrictVTablePointers &&
          cgm.getCodeGenOpts().OptimizationLevel > 0)
        llvm_unreachable("NYI");
      llvm_unreachable("NYI");
    }

    if (isTryBody)
      llvm_unreachable("NYI");
    else if (Body)
      (void)buildStmt(Body, /*useCurrentScope=*/true);
    else {
      assert(dtor->isImplicit() && "bodyless dtor not implicit");
      // nothing to do besides what's in the epilogue
    }
    // -fapple-kext must inline any call to this dtor into
    // the caller's body.
    if (getLangOpts().AppleKext)
      llvm_unreachable("NYI");

    break;
  }

  // Jump out through the epilogue cleanups.
  dtorEpilogue.ForceCleanup();

  // Exit the try if applicable.
  if (isTryBody)
    llvm_unreachable("NYI");
}

namespace {
[[maybe_unused]] mlir::Value
loadThisForDtorDelete(CIRGenFunction &cgf, const CXXDestructorDecl *dd) {
  if (Expr *thisArg = dd->getOperatorDeleteThisArg())
    return cgf.buildScalarExpr(thisArg);
  return cgf.LoadCXXThis();
}

/// Call the operator delete associated with the current destructor.
struct CallDtorDelete final : EHScopeStack::Cleanup {
  CallDtorDelete() = default;

  void Emit(CIRGenFunction &cgf, Flags flags) override {
    const CXXDestructorDecl *dtor = cast<CXXDestructorDecl>(cgf.CurCodeDecl);
    const CXXRecordDecl *classDecl = dtor->getParent();
    cgf.buildDeleteCall(dtor->getOperatorDelete(),
                        loadThisForDtorDelete(cgf, dtor),
                        cgf.getContext().getTagDeclType(classDecl));
  }
};
} // namespace

class DestroyField final : public EHScopeStack::Cleanup {
  const FieldDecl *field;
  CIRGenFunction::Destroyer *destroyer;
  bool useEHCleanupForArray;

public:
  DestroyField(const FieldDecl *field, CIRGenFunction::Destroyer *destroyer,
               bool useEHCleanupForArray)
      : field(field), destroyer(destroyer),
        useEHCleanupForArray(useEHCleanupForArray) {}

  void Emit(CIRGenFunction &cgf, Flags flags) override {
    // Find the address of the field.
    Address thisValue = cgf.LoadCXXThisAddress();
    QualType recordTy = cgf.getContext().getTagDeclType(field->getParent());
    LValue thisLv = cgf.makeAddrLValue(thisValue, recordTy);
    LValue lv = cgf.buildLValueForField(thisLv, field);
    assert(lv.isSimple());

    cgf.emitDestroy(lv.getAddress(), field->getType(), destroyer,
                    flags.isForNormalCleanup() && useEHCleanupForArray);
  }
};

/// Emit all code that comes at the end of class's destructor. This is to call
/// destructors on members and base classes in reverse order of their
/// construction.
///
/// For a deleting destructor, this also handles the case where a destroying
/// operator delete completely overrides the definition.
void CIRGenFunction::EnterDtorCleanups(const CXXDestructorDecl *dd,
                                       CXXDtorType dtorType) {
  assert((!dd->isTrivial() || dd->hasAttr<DLLExportAttr>()) &&
         "Should not emit dtor epilogue for non-exported trivial dtor!");

  // The deleting-destructor phase just needs to call the appropriate
  // operator delete that Sema picked up.
  if (dtorType == Dtor_Deleting) {
    assert(dd->getOperatorDelete() &&
           "operator delete missing - EnterDtorCleanups");
    if (CXXStructorImplicitParamValue) {
      llvm_unreachable("NYI");
    } else {
      if (dd->getOperatorDelete()->isDestroyingOperatorDelete()) {
        llvm_unreachable("NYI");
      } else {
        EHStack.pushCleanup<CallDtorDelete>(NormalAndEHCleanup);
      }
    }
    return;
  }

  const CXXRecordDecl *classDecl = dd->getParent();

  // Unions have no bases and do not call field destructors.
  if (classDecl->isUnion())
    return;

  // The complete-destructor phase just destructs all the virtual bases.
  if (dtorType == Dtor_Complete) {
    // Poison the vtable pointer such that access after the base
    // and member destructors are invoked is invalid.
    if (cgm.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
        SanOpts.has(SanitizerKind::Memory) && classDecl->getNumVBases() &&
        classDecl->isPolymorphic())
      assert(!MissingFeatures::sanitizeDtor());

    // We push them in the forward order so that they'll be popped in
    // the reverse order.
    for (const auto &base : classDecl->vbases()) {
      auto *baseClassDecl =
          cast<CXXRecordDecl>(base.getType()->castAs<RecordType>()->getDecl());

      if (baseClassDecl->hasTrivialDestructor()) {
        // Under SanitizeMemoryUseAfterDtor, poison the trivial base class
        // memory. For non-trival base classes the same is done in the class
        // destructor.
        assert(!MissingFeatures::sanitizeDtor());
      } else {
        EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, baseClassDecl,
                                          /*BaseIsVirtual*/ true);
      }
    }

    return;
  }

  assert(dtorType == Dtor_Base);
  // Poison the vtable pointer if it has no virtual bases, but inherits
  // virtual functions.
  if (cgm.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
      SanOpts.has(SanitizerKind::Memory) && !classDecl->getNumVBases() &&
      classDecl->isPolymorphic())
    assert(!MissingFeatures::sanitizeDtor());

  // Destroy non-virtual bases.
  for (const auto &base : classDecl->bases()) {
    // Ignore virtual bases.
    if (base.isVirtual())
      continue;

    CXXRecordDecl *baseClassDecl = base.getType()->getAsCXXRecordDecl();

    if (baseClassDecl->hasTrivialDestructor()) {
      if (cgm.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
          SanOpts.has(SanitizerKind::Memory) && !baseClassDecl->isEmpty())
        assert(!MissingFeatures::sanitizeDtor());
    } else {
      EHStack.pushCleanup<CallBaseDtor>(NormalAndEHCleanup, baseClassDecl,
                                        /*BaseIsVirtual*/ false);
    }
  }

  // Poison fields such that access after their destructors are
  // invoked, and before the base class destructor runs, is invalid.
  bool sanitizeFields = cgm.getCodeGenOpts().SanitizeMemoryUseAfterDtor &&
                        SanOpts.has(SanitizerKind::Memory);
  assert(!MissingFeatures::sanitizeDtor());

  // Destroy direct fields.
  for (const auto *field : classDecl->fields()) {
    if (sanitizeFields)
      assert(!MissingFeatures::sanitizeDtor());

    QualType type = field->getType();
    QualType::DestructionKind dtorKind = type.isDestructedType();
    if (!dtorKind)
      continue;

    // Anonymous union members do not have their destructors called.
    const RecordType *rt = type->getAsUnionType();
    if (rt && rt->getDecl()->isAnonymousStructOrUnion())
      continue;

    CleanupKind cleanupKind = getCleanupKind(dtorKind);
    EHStack.pushCleanup<DestroyField>(
        cleanupKind, field, getDestroyer(dtorKind), cleanupKind & EHCleanup);
  }

  if (sanitizeFields)
    assert(!MissingFeatures::sanitizeDtor());
}

namespace {
struct CallDelegatingCtorDtor final : EHScopeStack::Cleanup {
  const CXXDestructorDecl *dtor;
  Address addr;
  CXXDtorType type;

  CallDelegatingCtorDtor(const CXXDestructorDecl *d, Address addr,
                         CXXDtorType type)
      : dtor(d), addr(addr), type(type) {}

  void Emit(CIRGenFunction &cgf, Flags flags) override {
    // We are calling the destructor from within the constructor.
    // Therefore, "this" should have the expected type.
    QualType thisTy = dtor->getFunctionObjectParameterType();
    cgf.buildCXXDestructorCall(dtor, type, /*ForVirtualBase=*/false,
                               /*Delegating=*/true, addr, thisTy);
  }
};
} // end anonymous namespace

void CIRGenFunction::buildDelegatingCXXConstructorCall(
    const CXXConstructorDecl *ctor, const FunctionArgList &args) {
  assert(ctor->isDelegatingConstructor());

  Address thisPtr = LoadCXXThisAddress();

  AggValueSlot aggSlot = AggValueSlot::forAddr(
      thisPtr, Qualifiers(), AggValueSlot::IsDestructed,
      AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
      AggValueSlot::MayOverlap, AggValueSlot::IsNotZeroed,
      // Checks are made by the code that calls constructor.
      AggValueSlot::IsSanitizerChecked);

  buildAggExpr(ctor->init_begin()[0]->getInit(), aggSlot);

  const CXXRecordDecl *classDecl = ctor->getParent();
  if (cgm.getLangOpts().Exceptions && !classDecl->hasTrivialDestructor()) {
    CXXDtorType type =
        CurGD.getCtorType() == Ctor_Complete ? Dtor_Complete : Dtor_Base;

    EHStack.pushCleanup<CallDelegatingCtorDtor>(
        EHCleanup, classDecl->getDestructor(), thisPtr, type);
  }
}

void CIRGenFunction::buildCXXDestructorCall(const CXXDestructorDecl *dd,
                                            CXXDtorType type,
                                            bool forVirtualBase,
                                            bool delegating, Address This,
                                            QualType thisTy) {
  cgm.getCXXABI().buildDestructorCall(*this, dd, type, forVirtualBase,
                                      delegating, This, thisTy);
}

mlir::Value CIRGenFunction::GetVTTParameter(GlobalDecl gd, bool forVirtualBase,
                                            bool delegating) {
  if (!cgm.getCXXABI().NeedsVTTParameter(gd)) {
    // This constructor/destructor does not need a VTT parameter.
    return nullptr;
  }

  const CXXRecordDecl *rd = cast<CXXMethodDecl>(CurCodeDecl)->getParent();
  const CXXRecordDecl *base = cast<CXXMethodDecl>(gd.getDecl())->getParent();

  if (delegating) {
    llvm_unreachable("NYI");
  } else if (rd == base) {
    llvm_unreachable("NYI");
  } else {
    llvm_unreachable("NYI");
  }

  if (cgm.getCXXABI().NeedsVTTParameter(CurGD)) {
    llvm_unreachable("NYI");
  } else {
    llvm_unreachable("NYI");
  }
}

Address
CIRGenFunction::getAddressOfBaseClass(Address value,
                                      const CXXRecordDecl *derived,
                                      CastExpr::path_const_iterator pathBegin,
                                      CastExpr::path_const_iterator pathEnd,
                                      bool nullCheckValue, SourceLocation loc) {
  assert(pathBegin != pathEnd && "Base path should not be empty!");

  CastExpr::path_const_iterator start = pathBegin;
  const CXXRecordDecl *vBase = nullptr;

  // Sema has done some convenient canonicalization here: if the
  // access path involved any virtual steps, the conversion path will
  // *start* with a step down to the correct virtual base subobject,
  // and hence will not require any further steps.
  if ((*start)->isVirtual()) {
    llvm_unreachable("NYI: Cast to virtual base class");
  }

  // Compute the static offset of the ultimate destination within its
  // allocating subobject (the virtual base, if there is one, or else
  // the "complete" object that we see).
  CharUnits nonVirtualOffset = cgm.computeNonVirtualBaseClassOffset(
      vBase ? vBase : derived, start, pathEnd);

  // If there's a virtual step, we can sometimes "devirtualize" it.
  // For now, that's limited to when the derived type is final.
  // TODO: "devirtualize" this for accesses to known-complete objects.
  if (vBase && derived->hasAttr<FinalAttr>()) {
    const ASTRecordLayout &layout = getContext().getASTRecordLayout(derived);
    CharUnits vBaseOffset = layout.getVBaseClassOffset(vBase);
    nonVirtualOffset += vBaseOffset;
    vBase = nullptr; // we no longer have a virtual step
  }

  // Get the base pointer type.
  auto baseValueTy = convertType((pathEnd[-1])->getType());
  assert(!MissingFeatures::addressSpace());

  // If there is no virtual base, use cir.base_class_addr.  It takes care of
  // the adjustment and the null pointer check.
  if (!vBase) {
    if (sanitizePerformTypeCheck()) {
      llvm_unreachable("NYI: sanitizePerformTypeCheck");
    }
    return builder.createBaseClassAddr(getLoc(loc), value, baseValueTy,
                                       nonVirtualOffset.getQuantity(),
                                       /*assumeNotNull=*/not nullCheckValue);
  }

  // Conversion to a virtual base.  cir.base_class_addr can't handle this.
  // Generate the code to look up the address in the virtual table.

  llvm_unreachable("NYI: Cast to virtual base class");

  // This is just an outline of what the code might look like, since I can't
  // actually test it.
#if 0
  mlir::Value VirtualOffset = ...; // This is a dynamic expression.  Creating
                                   // it requires calling an ABI-specific
                                   // function.
  Value = ApplyNonVirtualAndVirtualOffset(getLoc(Loc), *this, Value,
                                          NonVirtualOffset, VirtualOffset,
                                          Derived, VBase);
  Value = builder.createElementBitCast(Value.getPointer().getLoc(), Value,
                                       BaseValueTy);
  if (sanitizePerformTypeCheck()) {
    // Do something here
  }
  if (NullCheckValue) {
    // Convert to 'derivedPtr == nullptr ? nullptr : basePtr'
  }
#endif

  return value;
}

// TODO(cir): this can be shared with LLVM codegen.
bool CIRGenFunction::shouldEmitVTableTypeCheckedLoad(const CXXRecordDecl *rd) {
  if (!cgm.getCodeGenOpts().WholeProgramVTables ||
      !cgm.HasHiddenLTOVisibility(rd))
    return false;

  if (cgm.getCodeGenOpts().VirtualFunctionElimination)
    return true;

  if (!SanOpts.has(SanitizerKind::CFIVCall) ||
      !cgm.getCodeGenOpts().SanitizeTrap.has(SanitizerKind::CFIVCall))
    return false;

  std::string typeName = rd->getQualifiedNameAsString();
  return !getContext().getNoSanitizeList().containsType(SanitizerKind::CFIVCall,
                                                        typeName);
}

void CIRGenFunction::buildTypeMetadataCodeForVCall(const CXXRecordDecl *rd,
                                                   mlir::Value vTable,
                                                   SourceLocation loc) {
  if (SanOpts.has(SanitizerKind::CFIVCall)) {
    llvm_unreachable("NYI");
  } else if (cgm.getCodeGenOpts().WholeProgramVTables &&
             // Don't insert type test assumes if we are forcing public
             // visibility.
             !cgm.AlwaysHasLTOVisibilityPublic(rd)) {
    llvm_unreachable("NYI");
  }
}

mlir::Value CIRGenFunction::getVTablePtr(mlir::Location loc, Address This,
                                         mlir::Type vTableTy,
                                         const CXXRecordDecl *rd) {
  Address vTablePtrSrc = builder.createElementBitCast(loc, This, vTableTy);
  auto vTable = builder.createLoad(loc, vTablePtrSrc);
  assert(!MissingFeatures::tbaa());

  if (cgm.getCodeGenOpts().OptimizationLevel > 0 &&
      cgm.getCodeGenOpts().StrictVTablePointers) {
    assert(!MissingFeatures::createInvariantGroup());
  }

  return vTable;
}

Address CIRGenFunction::buildCXXMemberDataPointerAddress(
    const Expr *e, Address base, mlir::Value memberPtr,
    const MemberPointerType *memberPtrType, LValueBaseInfo *baseInfo) {
  assert(!MissingFeatures::cxxABI());

  auto op = builder.createGetIndirectMember(getLoc(e->getSourceRange()),
                                            base.getPointer(), memberPtr);

  QualType memberType = memberPtrType->getPointeeType();
  CharUnits memberAlign = cgm.getNaturalTypeAlignment(memberType, baseInfo);
  memberAlign = cgm.getDynamicOffsetAlignment(
      base.getAlignment(), memberPtrType->getClass()->getAsCXXRecordDecl(),
      memberAlign);

  return Address(op, convertTypeForMem(memberPtrType->getPointeeType()),
                 memberAlign);
}

clang::CharUnits
CIRGenModule::getDynamicOffsetAlignment(clang::CharUnits actualBaseAlign,
                                        const clang::CXXRecordDecl *baseDecl,
                                        clang::CharUnits expectedTargetAlign) {
  // If the base is an incomplete type (which is, alas, possible with
  // member pointers), be pessimistic.
  if (!baseDecl->isCompleteDefinition())
    return std::min(actualBaseAlign, expectedTargetAlign);

  auto &baseLayout = getASTContext().getASTRecordLayout(baseDecl);
  CharUnits expectedBaseAlign = baseLayout.getNonVirtualAlignment();

  // If the class is properly aligned, assume the target offset is, too.
  //
  // This actually isn't necessarily the right thing to do --- if the
  // class is a complete object, but it's only properly aligned for a
  // base subobject, then the alignments of things relative to it are
  // probably off as well.  (Note that this requires the alignment of
  // the target to be greater than the NV alignment of the derived
  // class.)
  //
  // However, our approach to this kind of under-alignment can only
  // ever be best effort; after all, we're never going to propagate
  // alignments through variables or parameters.  Note, in particular,
  // that constructing a polymorphic type in an address that's less
  // than pointer-aligned will generally trap in the constructor,
  // unless we someday add some sort of attribute to change the
  // assumed alignment of 'this'.  So our goal here is pretty much
  // just to allow the user to explicitly say that a pointer is
  // under-aligned and then safely access its fields and vtables.
  if (actualBaseAlign >= expectedBaseAlign) {
    return expectedTargetAlign;
  }

  // Otherwise, we might be offset by an arbitrary multiple of the
  // actual alignment.  The correct adjustment is to take the min of
  // the two alignments.
  return std::min(actualBaseAlign, expectedTargetAlign);
}

/// Emit a loop to call a particular constructor for each of several members of
/// an array.
///
/// \param ctor the constructor to call for each element
/// \param arrayType the type of the array to initialize
/// \param arrayBegin an arrayType*
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CIRGenFunction::buildCXXAggrConstructorCall(
    const CXXConstructorDecl *ctor, const clang::ArrayType *arrayType,
    Address arrayBegin, const CXXConstructExpr *e, bool newPointerIsChecked,
    bool zeroInitialize) {
  QualType elementType;
  auto numElements = buildArrayLength(arrayType, elementType, arrayBegin);
  buildCXXAggrConstructorCall(ctor, numElements, arrayBegin, e,
                              newPointerIsChecked, zeroInitialize);
}

/// Emit a loop to call a particular constructor for each of several members of
/// an array.
///
/// \param ctor the constructor to call for each element
/// \param numElements the number of elements in the array;
///   may be zero
/// \param arrayBase a T*, where T is the type constructed by ctor
/// \param zeroInitialize true if each element should be
///   zero-initialized before it is constructed
void CIRGenFunction::buildCXXAggrConstructorCall(
    const CXXConstructorDecl *ctor, mlir::Value numElements, Address arrayBase,
    const CXXConstructExpr *e, bool newPointerIsChecked, bool zeroInitialize) {
  // It's legal for numElements to be zero.  This can happen both
  // dynamically, because x can be zero in 'new A[x]', and statically,
  // because of GCC extensions that permit zero-length arrays.  There
  // are probably legitimate places where we could assume that this
  // doesn't happen, but it's not clear that it's worth it.
  // llvm::BranchInst *zeroCheckBranch = nullptr;

  // Optimize for a constant count.
  auto constantCount =
      dyn_cast<mlir::cir::ConstantOp>(numElements.getDefiningOp());
  if (constantCount) {
    auto constIntAttr =
        mlir::dyn_cast<mlir::cir::IntAttr>(constantCount.getValue());
    // Just skip out if the constant count is zero.
    if (constIntAttr && constIntAttr.getUInt() == 0)
      return;
    // Otherwise, emit the check.
  } else {
    llvm_unreachable("NYI");
  }

  auto arrayTy =
      mlir::dyn_cast<mlir::cir::ArrayType>(arrayBase.getElementType());
  assert(arrayTy && "expected array type");
  auto elementType = arrayTy.getEltType();
  auto ptrToElmType = builder.getPointerTo(elementType);

  // Tradional LLVM codegen emits a loop here.
  // TODO(cir): Lower to a loop as part of LoweringPrepare.

  // The alignment of the base, adjusted by the size of a single element,
  // provides a conservative estimate of the alignment of every element.
  // (This assumes we never start tracking offsetted alignments.)
  //
  // Note that these are complete objects and so we don't need to
  // use the non-virtual size or alignment.
  QualType type = getContext().getTypeDeclType(ctor->getParent());
  CharUnits eltAlignment = arrayBase.getAlignment().alignmentOfArrayElement(
      getContext().getTypeSizeInChars(type));

  // Zero initialize the storage, if requested.
  if (zeroInitialize) {
    llvm_unreachable("NYI");
  }

  // C++ [class.temporary]p4:
  // There are two contexts in which temporaries are destroyed at a different
  // point than the end of the full-expression. The first context is when a
  // default constructor is called to initialize an element of an array.
  // If the constructor has one or more default arguments, the destruction of
  // every temporary created in a default argument expression is sequenced
  // before the construction of the next array element, if any.
  {
    RunCleanupsScope scope(*this);

    // Evaluate the constructor and its arguments in a regular
    // partial-destroy cleanup.
    if (getLangOpts().Exceptions &&
        !ctor->getParent()->hasTrivialDestructor()) {
      llvm_unreachable("NYI");
    }

    // Wmit the constructor call that will execute for every array element.
    builder.create<mlir::cir::ArrayCtor>(
        *currSrcLoc, arrayBase.getPointer(),
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto arg = b.getInsertionBlock()->addArgument(ptrToElmType, loc);
          Address curAddr = Address(arg, ptrToElmType, eltAlignment);
          auto currAVS = AggValueSlot::forAddr(
              curAddr, type.getQualifiers(), AggValueSlot::IsDestructed,
              AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
              AggValueSlot::DoesNotOverlap, AggValueSlot::IsNotZeroed,
              newPointerIsChecked ? AggValueSlot::IsSanitizerChecked
                                  : AggValueSlot::IsNotSanitizerChecked);
          buildCXXConstructorCall(ctor, Ctor_Complete, /*ForVirtualBase=*/false,
                                  /*Delegating=*/false, currAVS, e);
          builder.create<mlir::cir::YieldOp>(loc);
        });
  }

  if (constantCount.use_empty())
    constantCount.erase();
}
