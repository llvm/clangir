//===--- CIRGenExprCXX.cpp - Emit CIR Code for C++ expressions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of C++ expressions
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/MissingFeatures.h"
#include <CIRGenCXXABI.h>
#include <CIRGenFunction.h>
#include <CIRGenModule.h>
#include <CIRGenValue.h>

#include <clang/AST/DeclCXX.h>

using namespace cir;
using namespace clang;

namespace {
struct MemberCallInfo {
  RequiredArgs reqArgs;
  // Number of prefix arguments for the call. Ignores the `this` pointer.
  unsigned prefixSize;
};
} // namespace

static RValue buildNewDeleteCall(CIRGenFunction &cgf,
                                 const FunctionDecl *calleeDecl,
                                 const FunctionProtoType *calleeType,
                                 const CallArgList &args);

static MemberCallInfo
commonBuildCXXMemberOrOperatorCall(CIRGenFunction &cgf, const CXXMethodDecl *md,
                                   mlir::Value This, mlir::Value implicitParam,
                                   QualType implicitParamTy, const CallExpr *ce,
                                   CallArgList &args, CallArgList *rtlArgs) {
  assert(ce == nullptr || isa<CXXMemberCallExpr>(ce) ||
         isa<CXXOperatorCallExpr>(ce));
  assert(md->isInstance() &&
         "Trying to emit a member or operator call expr on a static method!");

  // Push the this ptr.
  const CXXRecordDecl *rd =
      cgf.CGM.getCXXABI().getThisArgumentTypeForMethod(md);
  args.add(RValue::get(This), cgf.getTypes().DeriveThisType(rd, md));

  // If there is an implicit parameter (e.g. VTT), emit it.
  if (implicitParam) {
    llvm_unreachable("NYI");
  }

  const auto *fpt = md->getType()->castAs<FunctionProtoType>();
  RequiredArgs required = RequiredArgs::forPrototypePlus(fpt, args.size());
  unsigned prefixSize = args.size() - 1;

  // Add the rest of the call args
  if (rtlArgs) {
    // Special case: if the caller emitted the arguments right-to-left already
    // (prior to emitting the *this argument), we're done. This happens for
    // assignment operators.
    args.addFrom(*rtlArgs);
  } else if (ce) {
    // Special case: skip first argument of CXXOperatorCall (it is "this").
    unsigned argsToSkip = isa<CXXOperatorCallExpr>(ce) ? 1 : 0;
    cgf.buildCallArgs(args, fpt, drop_begin(ce->arguments(), argsToSkip),
                      ce->getDirectCallee());
  } else {
    assert(
        fpt->getNumParams() == 0 &&
        "No CallExpr specified for function with non-zero number of arguments");
  }

  return {required, prefixSize};
}

RValue CIRGenFunction::buildCXXMemberOrOperatorCall(
    const CXXMethodDecl *md, const CIRGenCallee &callee,
    ReturnValueSlot returnValue, mlir::Value This, mlir::Value implicitParam,
    QualType implicitParamTy, const CallExpr *ce, CallArgList *rtlArgs) {

  const auto *fpt = md->getType()->castAs<FunctionProtoType>();
  CallArgList args;
  MemberCallInfo callInfo = commonBuildCXXMemberOrOperatorCall(
      *this, md, This, implicitParam, implicitParamTy, ce, args, rtlArgs);
  auto &fnInfo = CGM.getTypes().arrangeCXXMethodCall(
      args, fpt, callInfo.reqArgs, callInfo.prefixSize);
  assert((ce || currSrcLoc) && "expected source location");
  mlir::Location loc = ce ? getLoc(ce->getExprLoc()) : *currSrcLoc;
  return buildCall(fnInfo, callee, returnValue, args, nullptr,
                   ce && ce == MustTailCall, loc, ce);
}

// TODO(cir): this can be shared with LLVM codegen
static CXXRecordDecl *getCXXRecord(const Expr *e) {
  QualType t = e->getType();
  if (const PointerType *pTy = t->getAs<PointerType>())
    t = pTy->getPointeeType();
  const RecordType *ty = t->castAs<RecordType>();
  return cast<CXXRecordDecl>(ty->getDecl());
}

RValue
CIRGenFunction::buildCXXMemberPointerCallExpr(const CXXMemberCallExpr *e,
                                              ReturnValueSlot returnValue) {
  const BinaryOperator *bo =
      cast<BinaryOperator>(e->getCallee()->IgnoreParens());
  const Expr *baseExpr = bo->getLHS();
  const Expr *memFnExpr = bo->getRHS();

  const auto *mpt = memFnExpr->getType()->castAs<MemberPointerType>();
  const auto *fpt = mpt->getPointeeType()->castAs<FunctionProtoType>();

  // Emit the 'this' pointer.
  Address This = Address::invalid();
  if (bo->getOpcode() == BO_PtrMemI)
    This = buildPointerWithAlignment(baseExpr, nullptr, nullptr, KnownNonNull);
  else
    This = buildLValue(baseExpr).getAddress();

  buildTypeCheck(TCK_MemberCall, e->getExprLoc(), This.emitRawPointer(),
                 QualType(mpt->getClass(), 0));

  // Get the member function pointer.
  mlir::Value memFnPtr = buildScalarExpr(memFnExpr);

  // Resolve the member function pointer to the actual callee and adjust the
  // "this" pointer for call.
  auto loc = getLoc(e->getExprLoc());
  auto [CalleePtr, AdjustedThis] =
      builder.createGetMethod(loc, memFnPtr, This.getPointer());

  // Prepare the call arguments.
  CallArgList argsList;
  argsList.add(RValue::get(AdjustedThis), getContext().VoidPtrTy);
  buildCallArgs(argsList, fpt, e->arguments());

  RequiredArgs required = RequiredArgs::forPrototypePlus(fpt, 1);

  // Build the call.
  CIRGenCallee callee(fpt, CalleePtr.getDefiningOp());
  return buildCall(CGM.getTypes().arrangeCXXMethodCall(argsList, fpt, required,
                                                       /*PrefixSize=*/0),
                   callee, returnValue, argsList, nullptr, e == MustTailCall,
                   loc);
}

RValue CIRGenFunction::buildCXXMemberOrOperatorMemberCallExpr(
    const CallExpr *ce, const CXXMethodDecl *md, ReturnValueSlot returnValue,
    bool hasQualifier, NestedNameSpecifier *qualifier, bool isArrow,
    const Expr *base) {
  assert(isa<CXXMemberCallExpr>(ce) || isa<CXXOperatorCallExpr>(ce));

  // Compute the object pointer.
  bool canUseVirtualCall = md->isVirtual() && !hasQualifier;
  const CXXMethodDecl *devirtualizedMethod = nullptr;
  if (canUseVirtualCall &&
      md->getDevirtualizedMethod(base, getLangOpts().AppleKext)) {
    const CXXRecordDecl *bestDynamicDecl = base->getBestDynamicClassType();
    devirtualizedMethod = md->getCorrespondingMethodInClass(bestDynamicDecl);
    assert(devirtualizedMethod);
    const CXXRecordDecl *devirtualizedClass = devirtualizedMethod->getParent();
    const Expr *inner = base->IgnoreParenBaseCasts();
    if (devirtualizedMethod->getReturnType().getCanonicalType() !=
        md->getReturnType().getCanonicalType()) {
      // If the return types are not the same, this might be a case where more
      // code needs to run to compensate for it. For example, the derived
      // method might return a type that inherits form from the return
      // type of MD and has a prefix.
      // For now we just avoid devirtualizing these covariant cases.
      devirtualizedMethod = nullptr;
    } else if (getCXXRecord(inner) == devirtualizedClass) {
      // If the class of the Inner expression is where the dynamic method
      // is defined, build the this pointer from it.
      base = inner;
    } else if (getCXXRecord(base) != devirtualizedClass) {
      // If the method is defined in a class that is not the best dynamic
      // one or the one of the full expression, we would have to build
      // a derived-to-base cast to compute the correct this pointer, but
      // we don't have support for that yet, so do a virtual call.
      assert(!MissingFeatures::buildDerivedToBaseCastForDevirt());
      devirtualizedMethod = nullptr;
    }
  }

  bool trivialForCodegen =
      md->isTrivial() || (md->isDefaulted() && md->getParent()->isUnion());
  bool trivialAssignment =
      trivialForCodegen &&
      (md->isCopyAssignmentOperator() || md->isMoveAssignmentOperator()) &&
      !md->getParent()->mayInsertExtraPadding();
  (void)trivialAssignment;

  // C++17 demands that we evaluate the RHS of a (possibly-compound) assignment
  // operator before the LHS.
  CallArgList rtlArgStorage;
  CallArgList *rtlArgs = nullptr;
  LValue trivialAssignmentRhs;
  if (auto *oce = dyn_cast<CXXOperatorCallExpr>(ce)) {
    if (oce->isAssignmentOp()) {
      // See further note on TrivialAssignment, we don't handle this during
      // codegen, differently than LLVM, which early optimizes like this:
      //  if (TrivialAssignment) {
      //    TrivialAssignmentRHS = buildLValue(CE->getArg(1));
      //  } else {
      rtlArgs = &rtlArgStorage;
      buildCallArgs(*rtlArgs, md->getType()->castAs<FunctionProtoType>(),
                    drop_begin(ce->arguments(), 1), ce->getDirectCallee(),
                    /*ParamsToSkip*/ 0, EvaluationOrder::ForceRightToLeft);
    }
  }

  LValue This;
  if (isArrow) {
    LValueBaseInfo baseInfo;
    assert(!MissingFeatures::tbaa());
    Address thisValue = buildPointerWithAlignment(base, &baseInfo);
    This = makeAddrLValue(thisValue, base->getType(), baseInfo);
  } else {
    This = buildLValue(base);
  }

  if (const CXXConstructorDecl *ctor = dyn_cast<CXXConstructorDecl>(md)) {
    llvm_unreachable("NYI");
  }

  if (trivialForCodegen) {
    if (isa<CXXDestructorDecl>(md))
      return RValue::get(nullptr);

    if (trivialAssignment) {
      // From LLVM codegen:
      // We don't like to generate the trivial copy/move assignment operator
      // when it isn't necessary; just produce the proper effect here.
      // It's important that we use the result of EmitLValue here rather than
      // emitting call arguments, in order to preserve TBAA information from
      // the RHS.
      //
      // We don't early optimize like LLVM does:
      // LValue RHS = isa<CXXOperatorCallExpr>(CE) ? TrivialAssignmentRHS
      //                                           :
      //                                           buildLValue(*CE->arg_begin());
      // buildAggregateAssign(This, RHS, CE->getType());
      // return RValue::get(This.getPointer());
    } else {
      assert(md->getParent()->mayInsertExtraPadding() &&
             "unknown trivial member function");
    }
  }

  // Compute the function type we're calling
  const CXXMethodDecl *calleeDecl =
      devirtualizedMethod ? devirtualizedMethod : md;
  const CIRGenFunctionInfo *fInfo = nullptr;
  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(calleeDecl))
    llvm_unreachable("NYI");
  else
    fInfo = &CGM.getTypes().arrangeCXXMethodDeclaration(calleeDecl);

  auto ty = CGM.getTypes().GetFunctionType(*fInfo);

  // C++11 [class.mfct.non-static]p2:
  //   If a non-static member function of a class X is called for an object that
  //   is not of type X, or of a type derived from X, the behavior is undefined.
  SourceLocation callLoc;
  ASTContext &c = getContext();
  (void)c;
  if (ce)
    callLoc = ce->getExprLoc();

  SanitizerSet skippedChecks;
  if (const auto *cmce = dyn_cast<CXXMemberCallExpr>(ce)) {
    auto *ioa = cmce->getImplicitObjectArgument();
    auto isImplicitObjectCXXThis = isWrappedCXXThis(ioa);
    if (isImplicitObjectCXXThis)
      skippedChecks.set(SanitizerKind::Alignment, true);
    if (isImplicitObjectCXXThis || isa<DeclRefExpr>(ioa))
      skippedChecks.set(SanitizerKind::Null, true);
  }

  if (MissingFeatures::buildTypeCheck())
    llvm_unreachable("NYI");

  // C++ [class.virtual]p12:
  //   Explicit qualification with the scope operator (5.1) suppresses the
  //   virtual call mechanism.
  //
  // We also don't emit a virtual call if the base expression has a record type
  // because then we know what the type is.
  bool useVirtualCall = canUseVirtualCall && !devirtualizedMethod;

  if (const auto *dtor = dyn_cast<CXXDestructorDecl>(calleeDecl)) {
    llvm_unreachable("NYI");
  }

  // FIXME: Uses of 'MD' past this point need to be audited. We may need to use
  // 'CalleeDecl' instead.

  CIRGenCallee callee;
  if (useVirtualCall) {
    callee = CIRGenCallee::forVirtual(ce, md, This.getAddress(), ty);
  } else {
    if (SanOpts.has(SanitizerKind::CFINVCall)) {
      llvm_unreachable("NYI");
    }

    if (getLangOpts().AppleKext)
      llvm_unreachable("NYI");
    else if (!devirtualizedMethod)
      // TODO(cir): shouldn't this call getAddrOfCXXStructor instead?
      callee = CIRGenCallee::forDirect(CGM.GetAddrOfFunction(md, ty),
                                       GlobalDecl(md));
    else {
      callee = CIRGenCallee::forDirect(CGM.GetAddrOfFunction(md, ty),
                                       GlobalDecl(md));
    }
  }

  if (md->isVirtual()) {
    Address newThisAddr =
        CGM.getCXXABI().adjustThisArgumentForVirtualFunctionCall(
            *this, calleeDecl, This.getAddress(), useVirtualCall);
    This.setAddress(newThisAddr);
  }

  return buildCXXMemberOrOperatorCall(
      calleeDecl, callee, returnValue, This.getPointer(),
      /*ImplicitParam=*/nullptr, QualType(), ce, rtlArgs);
}

RValue
CIRGenFunction::buildCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *e,
                                               const CXXMethodDecl *md,
                                               ReturnValueSlot returnValue) {
  assert(md->isInstance() &&
         "Trying to emit a member call expr on a static method!");
  return buildCXXMemberOrOperatorMemberCallExpr(
      e, md, returnValue, /*HasQualifier=*/false, /*Qualifier=*/nullptr,
      /*IsArrow=*/false, e->getArg(0));
}

void CIRGenFunction::buildCXXConstructExpr(const CXXConstructExpr *e,
                                           AggValueSlot dest) {
  assert(!dest.isIgnored() && "Must have a destination!");
  const auto *cd = e->getConstructor();

  // If we require zero initialization before (or instead of) calling the
  // constructor, as can be the case with a non-user-provided default
  // constructor, emit the zero initialization now, unless destination is
  // already zeroed.
  if (e->requiresZeroInitialization() && !dest.isZeroed()) {
    switch (e->getConstructionKind()) {
    case CXXConstructionKind::Delegating:
    case CXXConstructionKind::Complete:
      buildNullInitialization(getLoc(e->getSourceRange()), dest.getAddress(),
                              e->getType());
      break;
    case CXXConstructionKind::VirtualBase:
    case CXXConstructionKind::NonVirtualBase:
      llvm_unreachable("NYI");
      break;
    }
  }

  // If this is a call to a trivial default constructor:
  // In LLVM: do nothing.
  // In CIR: emit as a regular call, other later passes should lower the
  // ctor call into trivial initialization.
  // if (CD->isTrivial() && CD->isDefaultConstructor())
  //  return;

  // Elide the constructor if we're constructing from a temporary
  if (getLangOpts().ElideConstructors && e->isElidable()) {
    // FIXME: This only handles the simplest case, where the source object is
    //        passed directly as the first argument to the constructor. This
    //        should also handle stepping through implicit casts and conversion
    //        sequences which involve two steps, with a conversion operator
    //        follwed by a converting constructor.
    const auto *srcObj = e->getArg(0);
    assert(srcObj->isTemporaryObject(getContext(), cd->getParent()));
    assert(
        getContext().hasSameUnqualifiedType(e->getType(), srcObj->getType()));
    buildAggExpr(srcObj, dest);
    return;
  }

  if (const ArrayType *arrayType = getContext().getAsArrayType(e->getType())) {
    buildCXXAggrConstructorCall(cd, arrayType, dest.getAddress(), e,
                                dest.isSanitizerChecked());
  } else {
    clang::CXXCtorType type = Ctor_Complete;
    bool forVirtualBase = false;
    bool delegating = false;

    switch (e->getConstructionKind()) {
    case CXXConstructionKind::Complete:
      type = Ctor_Complete;
      break;
    case CXXConstructionKind::Delegating:
      // We should be emitting a constructor; GlobalDecl will assert this
      type = CurGD.getCtorType();
      delegating = true;
      break;
    case CXXConstructionKind::VirtualBase:
      forVirtualBase = true;
      [[fallthrough]];
    case CXXConstructionKind::NonVirtualBase:
      type = Ctor_Base;
      break;
    }

    buildCXXConstructorCall(cd, type, forVirtualBase, delegating, dest, e);
  }
}

namespace {
/// The parameters to pass to a usual operator delete.
struct UsualDeleteParams {
  bool destroyingDelete = false;
  bool size = false;
  bool alignment = false;
};
} // namespace

// FIXME(cir): this should be shared with LLVM codegen
static UsualDeleteParams getUsualDeleteParams(const FunctionDecl *fd) {
  UsualDeleteParams params;

  const FunctionProtoType *fpt = fd->getType()->castAs<FunctionProtoType>();
  auto ai = fpt->param_type_begin(), ae = fpt->param_type_end();

  // The first argument is always a void*.
  ++ai;

  // The next parameter may be a std::destroying_delete_t.
  if (fd->isDestroyingOperatorDelete()) {
    params.destroyingDelete = true;
    assert(ai != ae);
    ++ai;
  }

  // Figure out what other parameters we should be implicitly passing.
  if (ai != ae && (*ai)->isIntegerType()) {
    params.size = true;
    ++ai;
  }

  if (ai != ae && (*ai)->isAlignValT()) {
    params.alignment = true;
    ++ai;
  }

  assert(ai == ae && "unexpected usual deallocation function parameter");
  return params;
}

static mlir::Value buildCXXNewAllocSize(CIRGenFunction &cgf,
                                        const CXXNewExpr *e,
                                        unsigned minElements,
                                        mlir::Value &numElements,
                                        mlir::Value &sizeWithoutCookie) {
  QualType type = e->getAllocatedType();

  if (!e->isArray()) {
    CharUnits typeSize = cgf.getContext().getTypeSizeInChars(type);
    sizeWithoutCookie = cgf.getBuilder().getConstant(
        cgf.getLoc(e->getSourceRange()),
        mlir::cir::IntAttr::get(cgf.SizeTy, typeSize.getQuantity()));
    return sizeWithoutCookie;
  }

  llvm_unreachable("NYI");
}

namespace {
/// A cleanup to call the given 'operator delete' function upon abnormal
/// exit from a new expression. Templated on a traits type that deals with
/// ensuring that the arguments dominate the cleanup if necessary.
template <typename Traits>
class CallDeleteDuringNew final : public EHScopeStack::Cleanup {
  /// Type used to hold llvm::Value*s.
  using ValueTy = typename Traits::ValueTy;
  /// Type used to hold RValues.
  using RValueTy = typename Traits::RValueTy;
  struct PlacementArg {
    RValueTy argValue;
    QualType argType;
  };

  unsigned numPlacementArgs : 31;
  unsigned passAlignmentToPlacementDelete : 1;
  const FunctionDecl *operatorDelete;
  ValueTy ptr;
  ValueTy allocSize;
  CharUnits allocAlign;

  PlacementArg *getPlacementArgs() {
    return reinterpret_cast<PlacementArg *>(this + 1);
  }

public:
  static size_t getExtraSize(size_t numPlacementArgs) {
    return numPlacementArgs * sizeof(PlacementArg);
  }

  CallDeleteDuringNew(size_t numPlacementArgs,
                      const FunctionDecl *operatorDelete, ValueTy ptr,
                      ValueTy allocSize, bool passAlignmentToPlacementDelete,
                      CharUnits allocAlign)
      : numPlacementArgs(numPlacementArgs),
        passAlignmentToPlacementDelete(passAlignmentToPlacementDelete),
        operatorDelete(operatorDelete), ptr(ptr), allocSize(allocSize),
        allocAlign(allocAlign) {}

  void setPlacementArg(unsigned i, RValueTy arg, QualType type) {
    assert(i < numPlacementArgs && "index out of range");
    getPlacementArgs()[i] = {arg, type};
  }

  void Emit(CIRGenFunction &cgf, Flags flags) override {
    const auto *fpt = operatorDelete->getType()->castAs<FunctionProtoType>();
    CallArgList deleteArgs;

    // The first argument is always a void* (or C* for a destroying operator
    // delete for class type C).
    deleteArgs.add(Traits::get(cgf, ptr), fpt->getParamType(0));

    // Figure out what other parameters we should be implicitly passing.
    UsualDeleteParams params;
    if (numPlacementArgs) {
      // A placement deallocation function is implicitly passed an alignment
      // if the placement allocation function was, but is never passed a size.
      params.alignment = passAlignmentToPlacementDelete;
    } else {
      // For a non-placement new-expression, 'operator delete' can take a
      // size and/or an alignment if it has the right parameters.
      params = getUsualDeleteParams(operatorDelete);
    }

    assert(!params.destroyingDelete &&
           "should not call destroying delete in a new-expression");

    // The second argument can be a std::size_t (for non-placement delete).
    if (params.size)
      deleteArgs.add(Traits::get(cgf, allocSize),
                     cgf.getContext().getSizeType());

    // The next (second or third) argument can be a std::align_val_t, which
    // is an enum whose underlying type is std::size_t.
    // FIXME: Use the right type as the parameter type. Note that in a call
    // to operator delete(size_t, ...), we may not have it available.
    if (params.alignment) {
      llvm_unreachable("NYI");
    }

    // Pass the rest of the arguments, which must match exactly.
    for (unsigned i = 0; i != numPlacementArgs; ++i) {
      auto arg = getPlacementArgs()[i];
      deleteArgs.add(Traits::get(cgf, arg.argValue), arg.argType);
    }

    // Call 'operator delete'.
    buildNewDeleteCall(cgf, operatorDelete, fpt, deleteArgs);
  }
};
} // namespace

/// Enter a cleanup to call 'operator delete' if the initializer in a
/// new-expression throws.
static void enterNewDeleteCleanup(CIRGenFunction &cgf, const CXXNewExpr *e,
                                  Address newPtr, mlir::Value allocSize,
                                  CharUnits allocAlign,
                                  const CallArgList &newArgs) {
  unsigned numNonPlacementArgs = e->passAlignment() ? 2 : 1;

  // If we're not inside a conditional branch, then the cleanup will
  // dominate and we can do the easier (and more efficient) thing.
  if (!cgf.isInConditionalBranch()) {
    struct DirectCleanupTraits {
      using ValueTy = mlir::Value;
      using RValueTy = RValue;
      static RValue get(CIRGenFunction &, ValueTy v) { return RValue::get(v); }
      static RValue get(CIRGenFunction &, RValueTy v) { return v; }
    };

    typedef CallDeleteDuringNew<DirectCleanupTraits> DirectCleanup;

    DirectCleanup *cleanup = cgf.EHStack.pushCleanupWithExtra<DirectCleanup>(
        EHCleanup, e->getNumPlacementArgs(), e->getOperatorDelete(),
        newPtr.getPointer(), allocSize, e->passAlignment(), allocAlign);
    for (unsigned i = 0, n = e->getNumPlacementArgs(); i != n; ++i) {
      auto &arg = newArgs[i + numNonPlacementArgs];
      cleanup->setPlacementArg(
          i, arg.getRValue(cgf, cgf.getLoc(e->getSourceRange())), arg.Ty);
    }

    return;
  }

  // Otherwise, we need to save all this stuff.
  DominatingValue<RValue>::saved_type savedNewPtr =
      DominatingValue<RValue>::save(cgf, RValue::get(newPtr.getPointer()));
  DominatingValue<RValue>::saved_type savedAllocSize =
      DominatingValue<RValue>::save(cgf, RValue::get(allocSize));

  struct ConditionalCleanupTraits {
    using ValueTy = DominatingValue<RValue>::saved_type;
    using RValueTy = DominatingValue<RValue>::saved_type;
    static RValue get(CIRGenFunction &cgf, ValueTy v) { return v.restore(cgf); }
  };
  typedef CallDeleteDuringNew<ConditionalCleanupTraits> ConditionalCleanup;

  ConditionalCleanup *cleanup =
      cgf.EHStack.pushCleanupWithExtra<ConditionalCleanup>(
          EHCleanup, e->getNumPlacementArgs(), e->getOperatorDelete(),
          savedNewPtr, savedAllocSize, e->passAlignment(), allocAlign);
  for (unsigned i = 0, n = e->getNumPlacementArgs(); i != n; ++i) {
    auto &arg = newArgs[i + numNonPlacementArgs];
    cleanup->setPlacementArg(
        i,
        DominatingValue<RValue>::save(
            cgf, arg.getRValue(cgf, cgf.getLoc(e->getSourceRange()))),
        arg.Ty);
  }

  cgf.initFullExprCleanup();
}

static void storeAnyExprIntoOneUnit(CIRGenFunction &cgf, const Expr *init,
                                    QualType allocType, Address newPtr,
                                    AggValueSlot::Overlap_t mayOverlap) {
  // FIXME: Refactor with buildExprAsInit.
  switch (cgf.getEvaluationKind(allocType)) {
  case TEK_Scalar:
    cgf.buildScalarInit(init, cgf.getLoc(init->getSourceRange()),
                        cgf.makeAddrLValue(newPtr, allocType), false);
    return;
  case TEK_Complex:
    llvm_unreachable("NYI");
    return;
  case TEK_Aggregate: {
    AggValueSlot slot = AggValueSlot::forAddr(
        newPtr, allocType.getQualifiers(), AggValueSlot::IsDestructed,
        AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
        mayOverlap, AggValueSlot::IsNotZeroed,
        AggValueSlot::IsSanitizerChecked);
    cgf.buildAggExpr(init, slot);
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}

static void buildNewInitializer(CIRGenFunction &cgf, const CXXNewExpr *e,
                                QualType elementType, mlir::Type elementTy,
                                Address newPtr, mlir::Value numElements,
                                mlir::Value allocSizeWithoutCookie) {
  assert(!MissingFeatures::generateDebugInfo());
  if (e->isArray()) {
    llvm_unreachable("NYI");
  } else if (const Expr *init = e->getInitializer()) {
    storeAnyExprIntoOneUnit(cgf, init, e->getAllocatedType(), newPtr,
                            AggValueSlot::DoesNotOverlap);
  }
}

static CharUnits calculateCookiePadding(CIRGenFunction &cgf,
                                        const CXXNewExpr *e) {
  if (!e->isArray())
    return CharUnits::Zero();

  // No cookie is required if the operator new[] being used is the
  // reserved placement operator new[].
  if (e->getOperatorNew()->isReservedGlobalPlacementOperator())
    return CharUnits::Zero();

  llvm_unreachable("NYI");
  // return CGF.CGM.getCXXABI().GetArrayCookieSize(E);
}

namespace {
/// Calls the given 'operator delete' on a single object.
struct CallObjectDelete final : EHScopeStack::Cleanup {
  mlir::Value ptr;
  const FunctionDecl *operatorDelete;
  QualType elementType;

  CallObjectDelete(mlir::Value ptr, const FunctionDecl *operatorDelete,
                   QualType elementType)
      : ptr(ptr), operatorDelete(operatorDelete), elementType(elementType) {}

  void Emit(CIRGenFunction &cgf, Flags flags) override {
    cgf.buildDeleteCall(operatorDelete, ptr, elementType);
  }
};
} // namespace

/// Emit the code for deleting a single object.
/// \return \c true if we started emitting UnconditionalDeleteBlock, \c false
/// if not.
static bool emitObjectDelete(CIRGenFunction &cgf, const CXXDeleteExpr *de,
                             Address ptr, QualType elementType) {
  // C++11 [expr.delete]p3:
  //   If the static type of the object to be deleted is different from its
  //   dynamic type, the static type shall be a base class of the dynamic type
  //   of the object to be deleted and the static type shall have a virtual
  //   destructor or the behavior is undefined.
  cgf.buildTypeCheck(CIRGenFunction::TCK_MemberCall, de->getExprLoc(),
                     ptr.getPointer(), elementType);

  const FunctionDecl *operatorDelete = de->getOperatorDelete();
  assert(!operatorDelete->isDestroyingOperatorDelete());

  // Find the destructor for the type, if applicable.  If the
  // destructor is virtual, we'll just emit the vcall and return.
  const CXXDestructorDecl *dtor = nullptr;
  if (const RecordType *rt = elementType->getAs<RecordType>()) {
    CXXRecordDecl *rd = cast<CXXRecordDecl>(rt->getDecl());
    if (rd->hasDefinition() && !rd->hasTrivialDestructor()) {
      dtor = rd->getDestructor();

      if (dtor->isVirtual()) {
        bool useVirtualCall = true;
        const Expr *base = de->getArgument();
        if (auto *devirtualizedDtor = dyn_cast_or_null<const CXXDestructorDecl>(
                dtor->getDevirtualizedMethod(
                    base, cgf.CGM.getLangOpts().AppleKext))) {
          useVirtualCall = false;
          const CXXRecordDecl *devirtualizedClass =
              devirtualizedDtor->getParent();
          if (declaresSameEntity(getCXXRecord(base), devirtualizedClass)) {
            // Devirtualized to the class of the base type (the type of the
            // whole expression).
            dtor = devirtualizedDtor;
          } else {
            // Devirtualized to some other type. Would need to cast the this
            // pointer to that type but we don't have support for that yet, so
            // do a virtual call. FIXME: handle the case where it is
            // devirtualized to the derived type (the type of the inner
            // expression) as in EmitCXXMemberOrOperatorMemberCallExpr.
            useVirtualCall = true;
          }
        }
        if (useVirtualCall) {
          llvm_unreachable("NYI");
          return false;
        }
      }
    }
  }

  // Make sure that we call delete even if the dtor throws.
  // This doesn't have to a conditional cleanup because we're going
  // to pop it off in a second.
  cgf.EHStack.pushCleanup<CallObjectDelete>(
      NormalAndEHCleanup, ptr.getPointer(), operatorDelete, elementType);

  if (dtor) {
    llvm_unreachable("NYI");
  } else if (auto lifetime = elementType.getObjCLifetime()) {
    switch (lifetime) {
    case Qualifiers::OCL_None:
    case Qualifiers::OCL_ExplicitNone:
    case Qualifiers::OCL_Autoreleasing:
      break;

    case Qualifiers::OCL_Strong:
      llvm_unreachable("NYI");
      break;

    case Qualifiers::OCL_Weak:
      llvm_unreachable("NYI");
      break;
    }
  }

  // In traditional LLVM codegen null checks are emitted to save a delete call.
  // In CIR we optimize for size by default, the null check should be added into
  // this function callers.
  assert(!MissingFeatures::emitNullCheckForDeleteCalls());

  cgf.PopCleanupBlock();
  return false;
}

void CIRGenFunction::buildCXXDeleteExpr(const CXXDeleteExpr *e) {
  const Expr *arg = e->getArgument();
  Address ptr = buildPointerWithAlignment(arg);

  // Null check the pointer.
  //
  // We could avoid this null check if we can determine that the object
  // destruction is trivial and doesn't require an array cookie; we can
  // unconditionally perform the operator delete call in that case. For now, we
  // assume that deleted pointers are null rarely enough that it's better to
  // keep the branch. This might be worth revisiting for a -O0 code size win.
  //
  // CIR note: emit the code size friendly by default for now, such as mentioned
  // in `EmitObjectDelete`.
  assert(!MissingFeatures::emitNullCheckForDeleteCalls());
  QualType deleteTy = e->getDestroyedType();

  // A destroying operator delete overrides the entire operation of the
  // delete expression.
  if (e->getOperatorDelete()->isDestroyingOperatorDelete()) {
    llvm_unreachable("NYI");
    return;
  }

  // We might be deleting a pointer to array.  If so, GEP down to the
  // first non-array element.
  // (this assumes that A(*)[3][7] is converted to [3 x [7 x %A]]*)
  if (deleteTy->isConstantArrayType()) {
    llvm_unreachable("NYI");
  }

  assert(convertTypeForMem(deleteTy) == ptr.getElementType());

  if (e->isArrayForm()) {
    llvm_unreachable("NYI");
  } else {
    (void)emitObjectDelete(*this, e, ptr, deleteTy);
  }
}

mlir::Value CIRGenFunction::buildCXXNewExpr(const CXXNewExpr *e) {
  // The element type being allocated.
  QualType allocType = getContext().getBaseElementType(e->getAllocatedType());

  // 1. Build a call to the allocation function.
  FunctionDecl *allocator = e->getOperatorNew();

  // If there is a brace-initializer, cannot allocate fewer elements than inits.
  unsigned minElements = 0;
  if (e->isArray() && e->hasInitializer()) {
    const InitListExpr *ile = dyn_cast<InitListExpr>(e->getInitializer());
    if (ile && ile->isStringLiteralInit())
      minElements =
          cast<ConstantArrayType>(ile->getType()->getAsArrayTypeUnsafe())
              ->getSize()
              .getZExtValue();
    else if (ile)
      minElements = ile->getNumInits();
  }

  mlir::Value numElements = nullptr;
  mlir::Value allocSizeWithoutCookie = nullptr;
  mlir::Value allocSize = buildCXXNewAllocSize(
      *this, e, minElements, numElements, allocSizeWithoutCookie);
  CharUnits allocAlign = getContext().getTypeAlignInChars(allocType);

  // Emit the allocation call.
  Address allocation = Address::invalid();
  CallArgList allocatorArgs;
  if (allocator->isReservedGlobalPlacementOperator()) {
    // If the allocator is a global placement operator, just
    // "inline" it directly.
    assert(e->getNumPlacementArgs() == 1);
    const Expr *arg = *e->placement_arguments().begin();

    LValueBaseInfo baseInfo;
    allocation = buildPointerWithAlignment(arg, &baseInfo);

    // The pointer expression will, in many cases, be an opaque void*.
    // In these cases, discard the computed alignment and use the
    // formal alignment of the allocated type.
    if (baseInfo.getAlignmentSource() != AlignmentSource::Decl)
      allocation = allocation.withAlignment(allocAlign);

    // Set up allocatorArgs for the call to operator delete if it's not
    // the reserved global operator.
    if (e->getOperatorDelete() &&
        !e->getOperatorDelete()->isReservedGlobalPlacementOperator()) {
      allocatorArgs.add(RValue::get(allocSize), getContext().getSizeType());
      allocatorArgs.add(RValue::get(allocation.getPointer()), arg->getType());
    }
  } else {
    const FunctionProtoType *allocatorType =
        allocator->getType()->castAs<FunctionProtoType>();
    unsigned paramsToSkip = 0;

    // The allocation size is the first argument.
    QualType sizeType = getContext().getSizeType();
    allocatorArgs.add(RValue::get(allocSize), sizeType);
    ++paramsToSkip;

    if (allocSize != allocSizeWithoutCookie) {
      llvm_unreachable("NYI");
    }

    // The allocation alignment may be passed as the second argument.
    if (e->passAlignment()) {
      llvm_unreachable("NYI");
    }

    // FIXME: Why do we not pass a CalleeDecl here?
    buildCallArgs(allocatorArgs, allocatorType, e->placement_arguments(),
                  /*AC*/
                  AbstractCallee(),
                  /*ParamsToSkip*/
                  paramsToSkip);
    RValue rv =
        buildNewDeleteCall(*this, allocator, allocatorType, allocatorArgs);

    // Set !heapallocsite metadata on the call to operator new.
    assert(!MissingFeatures::generateDebugInfo());

    // If this was a call to a global replaceable allocation function that does
    // not take an alignment argument, the allocator is known to produce storage
    // that's suitably aligned for any object that fits, up to a known
    // threshold. Otherwise assume it's suitably aligned for the allocated type.
    CharUnits allocationAlign = allocAlign;
    if (!e->passAlignment() &&
        allocator->isReplaceableGlobalAllocationFunction()) {
      auto &target = CGM.getASTContext().getTargetInfo();
      unsigned allocatorAlign = llvm::bit_floor(std::min<uint64_t>(
          target.getNewAlign(), getContext().getTypeSize(allocType)));
      allocationAlign = std::max(
          allocationAlign, getContext().toCharUnitsFromBits(allocatorAlign));
    }

    allocation = Address(rv.getScalarVal(), UInt8Ty, allocationAlign);
  }

  // Emit a null check on the allocation result if the allocation
  // function is allowed to return null (because it has a non-throwing
  // exception spec or is the reserved placement new) and we have an
  // interesting initializer will be running sanitizers on the initialization.
  bool nullCheck = e->shouldNullCheckAllocation() &&
                   (!allocType.isPODType(getContext()) || e->hasInitializer() ||
                    sanitizePerformTypeCheck());

  // The null-check means that the initializer is conditionally
  // evaluated.
  mlir::OpBuilder::InsertPoint ifBody, postIfBody, preIfBody;
  mlir::Value nullCmpResult;
  mlir::Location loc = getLoc(e->getSourceRange());

  if (nullCheck) {
    mlir::Value nullPtr =
        builder.getNullPtr(allocation.getPointer().getType(), loc);
    nullCmpResult = builder.createCompare(loc, mlir::cir::CmpOpKind::ne,
                                          allocation.getPointer(), nullPtr);
    preIfBody = builder.saveInsertionPoint();
    builder.create<mlir::cir::IfOp>(loc, nullCmpResult,
                                    /*withElseRegion=*/false,
                                    [&](mlir::OpBuilder &, mlir::Location) {
                                      ifBody = builder.saveInsertionPoint();
                                    });
    postIfBody = builder.saveInsertionPoint();
  }

  // Make sure the conditional evaluation uses the insertion
  // point right before the if check.
  mlir::OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
  if (ifBody.isSet()) {
    builder.setInsertionPointAfterValue(nullCmpResult);
    ip = builder.saveInsertionPoint();
  }
  ConditionalEvaluation conditional(ip);

  // All the actual work to be done should be placed inside the IfOp above,
  // so change the insertion point over there.
  if (ifBody.isSet()) {
    conditional.begin(*this);
    builder.restoreInsertionPoint(ifBody);
  }

  // If there's an operator delete, enter a cleanup to call it if an
  // exception is thrown.
  EHScopeStack::stable_iterator operatorDeleteCleanup;
  [[maybe_unused]] mlir::Operation *cleanupDominator = nullptr;
  if (e->getOperatorDelete() &&
      !e->getOperatorDelete()->isReservedGlobalPlacementOperator()) {
    enterNewDeleteCleanup(*this, e, allocation, allocSize, allocAlign,
                          allocatorArgs);
    operatorDeleteCleanup = EHStack.stable_begin();
    cleanupDominator =
        builder.create<mlir::cir::UnreachableOp>(getLoc(e->getSourceRange()))
            .getOperation();
  }

  assert((allocSize == allocSizeWithoutCookie) ==
         calculateCookiePadding(*this, e).isZero());
  if (allocSize != allocSizeWithoutCookie) {
    llvm_unreachable("NYI");
  }

  mlir::Type elementTy;
  Address result = Address::invalid();
  auto createCast = [&]() {
    elementTy = getTypes().convertTypeForMem(allocType);
    result = builder.createElementBitCast(getLoc(e->getSourceRange()),
                                          allocation, elementTy);
  };

  if (preIfBody.isSet()) {
    // Generate any cast before the if condition check on the null because the
    // result can be used after the if body and should dominate all potential
    // uses.
    mlir::OpBuilder::InsertionGuard guard(builder);
    assert(nullCmpResult && "expected");
    builder.setInsertionPointAfterValue(nullCmpResult);
    createCast();
  } else {
    createCast();
  }

  // Passing pointer through launder.invariant.group to avoid propagation of
  // vptrs information which may be included in previous type.
  // To not break LTO with different optimizations levels, we do it regardless
  // of optimization level.
  if (CGM.getCodeGenOpts().StrictVTablePointers &&
      allocator->isReservedGlobalPlacementOperator())
    llvm_unreachable("NYI");

  // Emit sanitizer checks for pointer value now, so that in the case of an
  // array it was checked only once and not at each constructor call. We may
  // have already checked that the pointer is non-null.
  // FIXME: If we have an array cookie and a potentially-throwing allocator,
  // we'll null check the wrong pointer here.
  SanitizerSet skippedChecks;
  skippedChecks.set(SanitizerKind::Null, nullCheck);
  buildTypeCheck(CIRGenFunction::TCK_ConstructorCall,
                 e->getAllocatedTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                 result.getPointer(), allocType, result.getAlignment(),
                 skippedChecks, numElements);

  buildNewInitializer(*this, e, allocType, elementTy, result, numElements,
                      allocSizeWithoutCookie);
  auto resultPtr = result.getPointer();
  if (e->isArray()) {
    llvm_unreachable("NYI");
  }

  // Deactivate the 'operator delete' cleanup if we finished
  // initialization.
  if (operatorDeleteCleanup.isValid()) {
    // FIXME: enable cleanupDominator above before implementing this.
    DeactivateCleanupBlock(operatorDeleteCleanup, cleanupDominator);
    if (cleanupDominator)
      cleanupDominator->erase();
  }

  if (nullCheck) {
    conditional.end(*this);
    // resultPtr is already updated in the first null check phase.

    // Reset insertion point to resume back to post ifOp.
    if (postIfBody.isSet()) {
      builder.create<mlir::cir::YieldOp>(loc);
      builder.restoreInsertionPoint(postIfBody);
    }
  }

  return resultPtr;
}

RValue CIRGenFunction::buildCXXDestructorCall(GlobalDecl dtor,
                                              const CIRGenCallee &callee,
                                              mlir::Value This, QualType thisTy,
                                              mlir::Value implicitParam,
                                              QualType implicitParamTy,
                                              const CallExpr *ce) {
  const CXXMethodDecl *dtorDecl = cast<CXXMethodDecl>(dtor.getDecl());

  assert(!thisTy.isNull());
  assert(thisTy->getAsCXXRecordDecl() == dtorDecl->getParent() &&
         "Pointer/Object mixup");

  LangAS srcAs = thisTy.getAddressSpace();
  LangAS dstAs = dtorDecl->getMethodQualifiers().getAddressSpace();
  if (srcAs != dstAs) {
    llvm_unreachable("NYI");
  }

  CallArgList args;
  commonBuildCXXMemberOrOperatorCall(*this, dtorDecl, This, implicitParam,
                                     implicitParamTy, ce, args, nullptr);
  assert((ce || dtor.getDecl()) && "expected source location provider");
  return buildCall(CGM.getTypes().arrangeCXXStructorDeclaration(dtor), callee,
                   ReturnValueSlot(), args, nullptr, ce && ce == MustTailCall,
                   ce ? getLoc(ce->getExprLoc())
                      : getLoc(dtor.getDecl()->getSourceRange()));
}

/// Emit a call to an operator new or operator delete function, as implicitly
/// created by new-expressions and delete-expressions.
static RValue buildNewDeleteCall(CIRGenFunction &cgf,
                                 const FunctionDecl *calleeDecl,
                                 const FunctionProtoType *calleeType,
                                 const CallArgList &args) {
  mlir::cir::CIRCallOpInterface callOrTryCall;
  auto calleePtr = cgf.CGM.GetAddrOfFunction(calleeDecl);
  CIRGenCallee callee =
      CIRGenCallee::forDirect(calleePtr, GlobalDecl(calleeDecl));
  RValue rv = cgf.buildCall(cgf.CGM.getTypes().arrangeFreeFunctionCall(
                                args, calleeType, /*ChainCall=*/false),
                            callee, ReturnValueSlot(), args, &callOrTryCall);

  /// C++1y [expr.new]p10:
  ///   [In a new-expression,] an implementation is allowed to omit a call
  ///   to a replaceable global allocation function.
  ///
  /// We model such elidable calls with the 'builtin' attribute.
  assert(!MissingFeatures::attributeBuiltin());
  return rv;
}

void CIRGenFunction::buildDeleteCall(const FunctionDecl *deleteFd,
                                     mlir::Value ptr, QualType deleteTy,
                                     mlir::Value numElements,
                                     CharUnits cookieSize) {
  assert((!numElements && cookieSize.isZero()) ||
         deleteFd->getOverloadedOperator() == OO_Array_Delete);

  const auto *deleteFTy = deleteFd->getType()->castAs<FunctionProtoType>();
  CallArgList deleteArgs;

  auto params = getUsualDeleteParams(deleteFd);
  const auto *paramTypeIt = deleteFTy->param_type_begin();

  // Pass the pointer itself.
  QualType argTy = *paramTypeIt++;
  mlir::Value deletePtr =
      builder.createBitcast(ptr.getLoc(), ptr, ConvertType(argTy));
  deleteArgs.add(RValue::get(deletePtr), argTy);

  // Pass the std::destroying_delete tag if present.
  mlir::Value destroyingDeleteTag{};
  if (params.destroyingDelete) {
    llvm_unreachable("NYI");
  }

  // Pass the size if the delete function has a size_t parameter.
  if (params.size) {
    QualType sizeType = *paramTypeIt++;
    CharUnits deleteTypeSize = getContext().getTypeSizeInChars(deleteTy);
    assert(SizeTy && "expected mlir::cir::IntType");
    auto size = builder.getConstInt(*currSrcLoc, ConvertType(sizeType),
                                    deleteTypeSize.getQuantity());

    // For array new, multiply by the number of elements.
    if (numElements) {
      // Uncomment upon adding testcase.
      // Size = builder.createMul(Size, NumElements);
      llvm_unreachable("NYI");
    }

    // If there is a cookie, add the cookie size.
    if (!cookieSize.isZero()) {
      // Uncomment upon adding testcase.
      // builder.createBinop(
      //     Size, mlir::cir::BinOpKind::Add,
      //     builder.getConstInt(*currSrcLoc, SizeTy,
      //     CookieSize.getQuantity()));
      llvm_unreachable("NYI");
    }

    deleteArgs.add(RValue::get(size), sizeType);
  }

  // Pass the alignment if the delete function has an align_val_t parameter.
  if (params.alignment) {
    llvm_unreachable("NYI");
  }

  assert(paramTypeIt == deleteFTy->param_type_end() &&
         "unknown parameter to usual delete function");

  // Emit the call to delete.
  buildNewDeleteCall(*this, deleteFd, deleteFTy, deleteArgs);

  // If call argument lowering didn't use the destroying_delete_t alloca,
  // remove it again.
  if (destroyingDeleteTag && destroyingDeleteTag.use_empty()) {
    llvm_unreachable("NYI"); // DestroyingDeleteTag->eraseFromParent();
  }
}

static mlir::Value buildDynamicCastToNull(CIRGenFunction &cgf,
                                          mlir::Location loc, QualType destTy) {
  mlir::Type destCirTy = cgf.ConvertType(destTy);
  assert(mlir::isa<mlir::cir::PointerType>(destCirTy) &&
         "result of dynamic_cast should be a ptr");

  mlir::Value nullPtrValue = cgf.getBuilder().getNullPtr(destCirTy, loc);

  if (!destTy->isPointerType()) {
    auto *currentRegion = cgf.getBuilder().getBlock()->getParent();
    /// C++ [expr.dynamic.cast]p9:
    ///   A failed cast to reference type throws std::bad_cast
    cgf.CGM.getCXXABI().buildBadCastCall(cgf, loc);

    // The call to bad_cast will terminate the current block. Create a new block
    // to hold any follow up code.
    cgf.getBuilder().createBlock(currentRegion, currentRegion->end());
  }

  return nullPtrValue;
}

mlir::Value CIRGenFunction::buildDynamicCast(Address thisAddr,
                                             const CXXDynamicCastExpr *dce) {
  auto loc = getLoc(dce->getSourceRange());

  CGM.buildExplicitCastExprType(dce, this);
  QualType destTy = dce->getTypeAsWritten();
  QualType srcTy = dce->getSubExpr()->getType();

  // C++ [expr.dynamic.cast]p7:
  //   If T is "pointer to cv void," then the result is a pointer to the most
  //   derived object pointed to by v.
  bool isDynCastToVoid = destTy->isVoidPointerType();
  bool isRefCast = destTy->isReferenceType();

  QualType srcRecordTy;
  QualType destRecordTy;
  if (isDynCastToVoid) {
    srcRecordTy = srcTy->getPointeeType();
    // No destRecordTy.
  } else if (const PointerType *destPTy = destTy->getAs<PointerType>()) {
    srcRecordTy = srcTy->castAs<PointerType>()->getPointeeType();
    destRecordTy = destPTy->getPointeeType();
  } else {
    srcRecordTy = srcTy;
    destRecordTy = destTy->castAs<ReferenceType>()->getPointeeType();
  }

  assert(srcRecordTy->isRecordType() && "source type must be a record type!");
  buildTypeCheck(TCK_DynamicOperation, dce->getExprLoc(), thisAddr.getPointer(),
                 srcRecordTy);

  if (dce->isAlwaysNull())
    return buildDynamicCastToNull(*this, loc, destTy);

  auto destCirTy = mlir::cast<mlir::cir::PointerType>(ConvertType(destTy));
  return CGM.getCXXABI().buildDynamicCast(*this, loc, srcRecordTy, destRecordTy,
                                          destCirTy, isRefCast, thisAddr);
}
