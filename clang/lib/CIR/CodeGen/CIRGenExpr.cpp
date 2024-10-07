//===--- CIRGenExpr.cpp - Emit LLVM Code from Expressions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes as CIR code.
//
//===----------------------------------------------------------------------===//
#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "CIRGenTBAA.h"
#include "CIRGenValue.h"
#include "EHScopeStack.h"
#include "TargetInfo.h"

#include "clang/AST/ExprCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include "llvm/ADT/StringExtras.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

static mlir::cir::FuncOp buildFunctionDeclPointer(CIRGenModule &cgm,
                                                  GlobalDecl gd) {
  const auto *fd = cast<FunctionDecl>(gd.getDecl());

  if (fd->hasAttr<WeakRefAttr>()) {
    mlir::Operation *aliasee = cgm.getWeakRefReference(fd);
    return dyn_cast<FuncOp>(aliasee);
  }

  auto v = cgm.GetAddrOfFunction(gd);

  return v;
}

static Address buildPreserveStructAccess(CIRGenFunction &cgf, LValue base,
                                         Address addr, const FieldDecl *field) {
  llvm_unreachable("NYI");
}

/// Get the address of a zero-sized field within a record. The resulting address
/// doesn't necessarily have the right type.
static Address buildAddrOfFieldStorage(CIRGenFunction &cgf, Address base,
                                       const FieldDecl *field,
                                       llvm::StringRef fieldName,
                                       unsigned fieldIndex) {
  if (field->isZeroSize(cgf.getContext()))
    llvm_unreachable("NYI");

  auto loc = cgf.getLoc(field->getLocation());

  auto fieldType = cgf.convertType(field->getType());
  auto fieldPtr =
      mlir::cir::PointerType::get(cgf.getBuilder().getContext(), fieldType);
  // For most cases fieldName is the same as field->getName() but for lambdas,
  // which do not currently carry the name, so it can be passed down from the
  // CaptureStmt.
  auto memberAddr = cgf.getBuilder().createGetMember(
      loc, fieldPtr, base.getPointer(), fieldName, fieldIndex);

  // Retrieve layout information, compute alignment and return the final
  // address.
  const RecordDecl *rec = field->getParent();
  auto &layout = cgf.CGM.getTypes().getCIRGenRecordLayout(rec);
  unsigned idx = layout.getCIRFieldNo(field);
  auto offset = CharUnits::fromQuantity(layout.getCIRType().getElementOffset(
      cgf.CGM.getDataLayout().layout, idx));
  auto addr =
      Address(memberAddr, base.getAlignment().alignmentAtOffset(offset));
  return addr;
}

static bool hasAnyVptr(const QualType type, const ASTContext &context) {
  const auto *rd = type.getTypePtr()->getAsCXXRecordDecl();
  if (!rd)
    return false;

  if (rd->isDynamicClass())
    return true;

  for (const auto &base : rd->bases())
    if (hasAnyVptr(base.getType(), context))
      return true;

  for (const FieldDecl *field : rd->fields())
    if (hasAnyVptr(field->getType(), context))
      return true;

  return false;
}

static Address buildPointerWithAlignment(const Expr *expr,
                                         LValueBaseInfo *baseInfo,
                                         TBAAAccessInfo *tbaaInfo,
                                         KnownNonNull_t isKnownNonNull,
                                         CIRGenFunction &cgf) {
  // We allow this with ObjC object pointers because of fragile ABIs.
  assert(expr->getType()->isPointerType() ||
         expr->getType()->isObjCObjectPointerType());
  expr = expr->IgnoreParens();

  // Casts:
  if (const CastExpr *ce = dyn_cast<CastExpr>(expr)) {
    if (const auto *ece = dyn_cast<ExplicitCastExpr>(ce))
      cgf.CGM.buildExplicitCastExprType(ece, &cgf);

    switch (ce->getCastKind()) {
    default: {
      llvm::errs() << ce->getCastKindName() << "\n";
      assert(0 && "not implemented");
    }
    // Non-converting casts (but not C's implicit conversion from void*).
    case CK_BitCast:
    case CK_NoOp:
    case CK_AddressSpaceConversion:
      if (const auto *ptrTy =
              ce->getSubExpr()->getType()->getAs<clang::PointerType>()) {
        if (ptrTy->getPointeeType()->isVoidType())
          break;
        assert(!MissingFeatures::tbaa());

        LValueBaseInfo innerBaseInfo;
        Address addr = cgf.buildPointerWithAlignment(
            ce->getSubExpr(), &innerBaseInfo, tbaaInfo, isKnownNonNull);
        if (baseInfo)
          *baseInfo = innerBaseInfo;

        if (isa<ExplicitCastExpr>(ce)) {
          assert(!MissingFeatures::tbaa());
          LValueBaseInfo targetTypeBaseInfo;

          CharUnits align = cgf.CGM.getNaturalPointeeTypeAlignment(
              expr->getType(), &targetTypeBaseInfo);

          // If the source l-value is opaque, honor the alignment of the
          // casted-to type.
          if (innerBaseInfo.getAlignmentSource() != AlignmentSource::Decl) {
            if (baseInfo)
              baseInfo->mergeForCast(targetTypeBaseInfo);
            addr = Address(addr.getPointer(), addr.getElementType(), align,
                           isKnownNonNull);
          }
        }

        if (cgf.SanOpts.has(SanitizerKind::CFIUnrelatedCast) &&
            ce->getCastKind() == CK_BitCast) {
          if (const auto *pt = expr->getType()->getAs<clang::PointerType>())
            llvm_unreachable("NYI");
        }

        auto elemTy =
            cgf.getTypes().convertTypeForMem(expr->getType()->getPointeeType());
        addr = cgf.getBuilder().createElementBitCast(
            cgf.getLoc(expr->getSourceRange()), addr, elemTy);
        if (ce->getCastKind() == CK_AddressSpaceConversion) {
          assert(!MissingFeatures::addressSpace());
          llvm_unreachable("NYI");
        }
        return addr;
      }
      break;

    // Nothing to do here...
    case CK_LValueToRValue:
    case CK_NullToPointer:
    case CK_IntegralToPointer:
      break;

    // Array-to-pointer decay. TODO(cir): BaseInfo and TBAAInfo.
    case CK_ArrayToPointerDecay:
      return cgf.buildArrayToPointerDecay(ce->getSubExpr());

    case CK_UncheckedDerivedToBase:
    case CK_DerivedToBase: {
      // TODO: Support accesses to members of base classes in TBAA. For now, we
      // conservatively pretend that the complete object is of the base class
      // type.
      assert(!MissingFeatures::tbaa());
      Address addr = cgf.buildPointerWithAlignment(ce->getSubExpr(), baseInfo);
      const auto *derived =
          ce->getSubExpr()->getType()->getPointeeCXXRecordDecl();
      return cgf.getAddressOfBaseClass(
          addr, derived, ce->path_begin(), ce->path_end(),
          cgf.shouldNullCheckClassCastValue(ce), ce->getExprLoc());
    }
    }
  }

  // Unary &.
  if (const UnaryOperator *uo = dyn_cast<UnaryOperator>(expr)) {
    // TODO(cir): maybe we should use cir.unary for pointers here instead.
    if (uo->getOpcode() == UO_AddrOf) {
      LValue lv = cgf.buildLValue(uo->getSubExpr());
      if (baseInfo)
        *baseInfo = lv.getBaseInfo();
      assert(!MissingFeatures::tbaa());
      return lv.getAddress();
    }
  }

  // std::addressof and variants.
  if (auto *call = dyn_cast<CallExpr>(expr)) {
    switch (call->getBuiltinCallee()) {
    default:
      break;
    case Builtin::BIaddressof:
    case Builtin::BI__addressof:
    case Builtin::BI__builtin_addressof: {
      llvm_unreachable("NYI");
    }
    }
  }

  // TODO: conditional operators, comma.

  // Otherwise, use the alignment of the type.
  return cgf.makeNaturalAddressForPointer(
      cgf.buildScalarExpr(expr), expr->getType()->getPointeeType(), CharUnits(),
      /*ForPointeeType=*/true, baseInfo, tbaaInfo, isKnownNonNull);
}

/// Helper method to check if the underlying ABI is AAPCS
static bool isAAPCS(const TargetInfo &targetInfo) {
  return targetInfo.getABI().starts_with("aapcs");
}

Address CIRGenFunction::getAddrOfBitFieldStorage(LValue base,
                                                 const FieldDecl *field,
                                                 mlir::Type fieldType,
                                                 unsigned index) {
  if (index == 0)
    return base.getAddress();
  auto loc = getLoc(field->getLocation());
  auto fieldPtr =
      mlir::cir::PointerType::get(getBuilder().getContext(), fieldType);
  auto sea = getBuilder().createGetMember(loc, fieldPtr, base.getPointer(),
                                          field->getName(), index);
  return Address(sea, CharUnits::One());
}

static bool useVolatileForBitField(const CIRGenModule &cgm, LValue base,
                                   const CIRGenBitFieldInfo &info,
                                   const FieldDecl *field) {
  return isAAPCS(cgm.getTarget()) && cgm.getCodeGenOpts().AAPCSBitfieldWidth &&
         info.VolatileStorageSize != 0 &&
         field->getType()
             .withCVRQualifiers(base.getVRQualifiers())
             .isVolatileQualified();
}

LValue CIRGenFunction::buildLValueForBitField(LValue base,
                                              const FieldDecl *field) {

  LValueBaseInfo baseInfo = base.getBaseInfo();
  const RecordDecl *rec = field->getParent();
  auto &layout = CGM.getTypes().getCIRGenRecordLayout(field->getParent());
  auto &info = layout.getBitFieldInfo(field);
  auto useVolatile = useVolatileForBitField(CGM, base, info, field);
  unsigned idx = layout.getCIRFieldNo(field);

  if (useVolatile ||
      (IsInPreservedAIRegion ||
       (getDebugInfo() && rec->hasAttr<BPFPreserveAccessIndexAttr>()))) {
    llvm_unreachable("NYI");
  }

  Address addr = getAddrOfBitFieldStorage(base, field, info.StorageType, idx);

  auto loc = getLoc(field->getLocation());
  if (addr.getElementType() != info.StorageType)
    addr = builder.createElementBitCast(loc, addr, info.StorageType);

  QualType fieldType =
      field->getType().withCVRQualifiers(base.getVRQualifiers());
  assert(!MissingFeatures::tbaa() && "NYI TBAA for bit fields");
  LValueBaseInfo fieldBaseInfo(baseInfo.getAlignmentSource());
  return LValue::MakeBitfield(addr, info, fieldType, fieldBaseInfo,
                              TBAAAccessInfo());
}

LValue CIRGenFunction::buildLValueForField(LValue base,
                                           const FieldDecl *field) {
  LValueBaseInfo baseInfo = base.getBaseInfo();

  if (field->isBitField())
    return buildLValueForBitField(base, field);

  // Fields of may-alias structures are may-alais themselves.
  // FIXME: this hould get propagated down through anonymous structs and unions.
  QualType fieldType = field->getType();
  const RecordDecl *rec = field->getParent();
  AlignmentSource baseAlignSource = baseInfo.getAlignmentSource();
  LValueBaseInfo fieldBaseInfo(getFieldAlignmentSource(baseAlignSource));
  if (MissingFeatures::tbaa() || rec->hasAttr<MayAliasAttr>() ||
      fieldType->isVectorType()) {
    assert(!MissingFeatures::tbaa() && "NYI");
  } else if (rec->isUnion()) {
    assert(!MissingFeatures::tbaa() && "NYI");
  } else {
    // If no base type been assigned for the base access, then try to generate
    // one for this base lvalue.
    assert(!MissingFeatures::tbaa() && "NYI");
  }

  Address addr = base.getAddress();
  if (auto *classDef = dyn_cast<CXXRecordDecl>(rec)) {
    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        classDef->isDynamicClass()) {
      llvm_unreachable("NYI");
    }
  }

  unsigned recordCvr = base.getVRQualifiers();
  if (rec->isUnion()) {
    // NOTE(cir): the element to be loaded/stored need to type-match the
    // source/destination, so we emit a GetMemberOp here.
    llvm::StringRef fieldName = field->getName();
    unsigned fieldIndex = field->getFieldIndex();
    if (CGM.LambdaFieldToName.count(field))
      fieldName = CGM.LambdaFieldToName[field];
    addr = buildAddrOfFieldStorage(*this, addr, field, fieldName, fieldIndex);

    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        hasAnyVptr(fieldType, getContext()))
      // Because unions can easily skip invariant.barriers, we need to add
      // a barrier every time CXXRecord field with vptr is referenced.
      assert(!MissingFeatures::createInvariantGroup());

    if (IsInPreservedAIRegion ||
        (getDebugInfo() && rec->hasAttr<BPFPreserveAccessIndexAttr>())) {
      assert(!MissingFeatures::generateDebugInfo());
    }

    if (fieldType->isReferenceType())
      llvm_unreachable("NYI");
  } else {
    if (!IsInPreservedAIRegion &&
        (!getDebugInfo() || !rec->hasAttr<BPFPreserveAccessIndexAttr>())) {
      llvm::StringRef fieldName = field->getName();
      auto &layout = CGM.getTypes().getCIRGenRecordLayout(field->getParent());
      unsigned fieldIndex = layout.getCIRFieldNo(field);

      if (CGM.LambdaFieldToName.count(field))
        fieldName = CGM.LambdaFieldToName[field];
      addr = buildAddrOfFieldStorage(*this, addr, field, fieldName, fieldIndex);
    } else
      // Remember the original struct field index
      addr = buildPreserveStructAccess(*this, base, addr, field);
  }

  // If this is a reference field, load the reference right now.
  if (fieldType->isReferenceType()) {
    assert(!MissingFeatures::tbaa());
    LValue refLVal = makeAddrLValue(addr, fieldType, fieldBaseInfo);
    if (recordCvr & Qualifiers::Volatile)
      refLVal.getQuals().addVolatile();
    addr = buildLoadOfReference(refLVal, getLoc(field->getSourceRange()),
                                &fieldBaseInfo);

    // Qualifiers on the struct don't apply to the referencee.
    recordCvr = 0;
    fieldType = fieldType->getPointeeType();
  }

  // Make sure that the address is pointing to the right type. This is critical
  // for both unions and structs. A union needs a bitcast, a struct element will
  // need a bitcast if the CIR type laid out doesn't match the desired type.
  // TODO(CIR): CodeGen requires a bitcast here for unions or for structs where
  // the LLVM type doesn't match the desired type. No idea when the latter might
  // occur, though.

  if (field->hasAttr<AnnotateAttr>())
    llvm_unreachable("NYI");

  if (MissingFeatures::tbaa())
    // Next line should take a TBAA object
    llvm_unreachable("NYI");
  LValue lv = makeAddrLValue(addr, fieldType, fieldBaseInfo);
  lv.getQuals().addCVRQualifiers(recordCvr);

  // __weak attribute on a field is ignored.
  if (lv.getQuals().getObjCGCAttr() == Qualifiers::Weak)
    llvm_unreachable("NYI");

  return lv;
}

LValue CIRGenFunction::buildLValueForFieldInitialization(
    LValue base, const clang::FieldDecl *field, llvm::StringRef fieldName) {
  QualType fieldType = field->getType();

  if (!fieldType->isReferenceType())
    return buildLValueForField(base, field);

  auto &layout = CGM.getTypes().getCIRGenRecordLayout(field->getParent());
  unsigned fieldIndex = layout.getCIRFieldNo(field);

  Address v = buildAddrOfFieldStorage(*this, base.getAddress(), field,
                                      fieldName, fieldIndex);

  // Make sure that the address is pointing to the right type.
  auto memTy = getTypes().convertTypeForMem(fieldType);
  v = builder.createElementBitCast(getLoc(field->getSourceRange()), v, memTy);

  // TODO: Generate TBAA information that describes this access as a structure
  // member access and not just an access to an object of the field's type. This
  // should be similar to what we do in EmitLValueForField().
  LValueBaseInfo baseInfo = base.getBaseInfo();
  AlignmentSource fieldAlignSource = baseInfo.getAlignmentSource();
  LValueBaseInfo fieldBaseInfo(getFieldAlignmentSource(fieldAlignSource));
  assert(!MissingFeatures::tbaa() && "NYI");
  return makeAddrLValue(v, fieldType, fieldBaseInfo);
}

LValue
CIRGenFunction::buildCompoundLiteralLValue(const CompoundLiteralExpr *e) {
  if (e->isFileScope()) {
    llvm_unreachable("NYI");
  }

  if (e->getType()->isVariablyModifiedType()) {
    llvm_unreachable("NYI");
  }

  Address declPtr = CreateMemTemp(e->getType(), getLoc(e->getSourceRange()),
                                  ".compoundliteral");
  const Expr *initExpr = e->getInitializer();
  LValue result = makeAddrLValue(declPtr, e->getType(), AlignmentSource::Decl);

  buildAnyExprToMem(initExpr, declPtr, e->getType().getQualifiers(),
                    /*Init*/ true);

  // Block-scope compound literals are destroyed at the end of the enclosing
  // scope in C.
  if (!getLangOpts().CPlusPlus)
    if (QualType::DestructionKind dtorKind = e->getType().isDestructedType())
      llvm_unreachable("NYI");

  return result;
}

// Detect the unusual situation where an inline version is shadowed by a
// non-inline version. In that case we should pick the external one
// everywhere. That's GCC behavior too.
static bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *fd) {
  for (const FunctionDecl *pd = fd; pd; pd = pd->getPreviousDecl())
    if (!pd->isInlineBuiltinDeclaration())
      return false;
  return true;
}

static CIRGenCallee buildDirectCallee(CIRGenModule &cgm, GlobalDecl gd) {
  const auto *fd = cast<FunctionDecl>(gd.getDecl());

  if (auto builtinID = fd->getBuiltinID()) {
    std::string noBuiltinFd = ("no-builtin-" + fd->getName()).str();
    std::string noBuiltins = "no-builtins";

    auto *a = fd->getAttr<AsmLabelAttr>();
    StringRef ident = a ? a->getLabel() : fd->getName();
    std::string fdInlineName = (ident + ".inline").str();

    auto &cgf = *cgm.getCurrCIRGenFun();
    bool isPredefinedLibFunction =
        cgm.getASTContext().BuiltinInfo.isPredefinedLibFunction(builtinID);
    bool hasAttributeNoBuiltin = false;
    assert(!MissingFeatures::attributeNoBuiltin() && "NYI");
    // bool HasAttributeNoBuiltin =
    //     CGF.CurFn->getAttributes().hasFnAttr(NoBuiltinFD) ||
    //     CGF.CurFn->getAttributes().hasFnAttr(NoBuiltins);

    // When directing calling an inline builtin, call it through it's mangled
    // name to make it clear it's not the actual builtin.
    auto fn = cast<mlir::cir::FuncOp>(cgf.CurFn);
    if (fn.getName() != fdInlineName && onlyHasInlineBuiltinDeclaration(fd)) {
      assert(0 && "NYI");
    }

    // Replaceable builtins provide their own implementation of a builtin. If we
    // are in an inline builtin implementation, avoid trivial infinite
    // recursion. Honor __attribute__((no_builtin("foo"))) or
    // __attribute__((no_builtin)) on the current function unless foo is
    // not a predefined library function which means we must generate the
    // builtin no matter what.
    else if (!isPredefinedLibFunction || !hasAttributeNoBuiltin)
      return CIRGenCallee::forBuiltin(builtinID, fd);
  }

  auto calleePtr = buildFunctionDeclPointer(cgm, gd);

  assert(!cgm.getLangOpts().CUDA && "NYI");

  return CIRGenCallee::forDirect(calleePtr, gd);
}

// TODO: this can also be abstrated into common AST helpers
bool CIRGenFunction::hasBooleanRepresentation(QualType ty) {

  if (ty->isBooleanType())
    return true;

  if (const EnumType *et = ty->getAs<EnumType>())
    return et->getDecl()->getIntegerType()->isBooleanType();

  if (const AtomicType *at = ty->getAs<AtomicType>())
    return hasBooleanRepresentation(at->getValueType());

  return false;
}

CIRGenCallee CIRGenFunction::buildCallee(const clang::Expr *e) {
  e = e->IgnoreParens();

  // Look through function-to-pointer decay.
  if (const auto *ice = dyn_cast<ImplicitCastExpr>(e)) {
    if (ice->getCastKind() == CK_FunctionToPointerDecay ||
        ice->getCastKind() == CK_BuiltinFnToFnPtr) {
      return buildCallee(ice->getSubExpr());
    }
    // Resolve direct calls.
  } else if (const auto *dre = dyn_cast<DeclRefExpr>(e)) {
    const auto *fd = dyn_cast<FunctionDecl>(dre->getDecl());
    assert(fd &&
           "DeclRef referring to FunctionDecl only thing supported so far");
    return buildDirectCallee(CGM, fd);
  } else if (const auto *me = dyn_cast<MemberExpr>(e)) {
    if (auto *fd = dyn_cast<FunctionDecl>(me->getMemberDecl())) {
      buildIgnoredExpr(me->getBase());
      return buildDirectCallee(CGM, fd);
    }
  }

  assert(!dyn_cast<SubstNonTypeTemplateParmExpr>(e) && "NYI");
  assert(!dyn_cast<CXXPseudoDestructorExpr>(e) && "NYI");

  // Otherwise, we have an indirect reference.
  mlir::Value calleePtr;
  QualType functionType;
  if (const auto *ptrType = e->getType()->getAs<clang::PointerType>()) {
    calleePtr = buildScalarExpr(e);
    functionType = ptrType->getPointeeType();
  } else {
    functionType = e->getType();
    calleePtr = buildLValue(e).getPointer();
  }
  assert(functionType->isFunctionType());

  GlobalDecl gd;
  if (const auto *vd =
          dyn_cast_or_null<VarDecl>(e->getReferencedDeclOfCallee()))
    gd = GlobalDecl(vd);

  CIRGenCalleeInfo calleeInfo(functionType->getAs<FunctionProtoType>(), gd);
  CIRGenCallee callee(calleeInfo, calleePtr.getDefiningOp());
  return callee;

  assert(false && "Nothing else supported yet!");
}

mlir::Value CIRGenFunction::buildToMemory(mlir::Value value, QualType ty) {
  // Bool has a different representation in memory than in registers.
  return value;
}

void CIRGenFunction::buildStoreOfScalar(mlir::Value value, LValue lvalue) {
  // TODO: constant matrix type, no init, non temporal, TBAA
  buildStoreOfScalar(value, lvalue.getAddress(), lvalue.isVolatile(),
                     lvalue.getType(), lvalue.getBaseInfo(),
                     lvalue.getTBAAInfo(), false, false);
}

void CIRGenFunction::buildStoreOfScalar(mlir::Value value, Address addr,
                                        bool isVolatile, QualType ty,
                                        LValueBaseInfo baseInfo,
                                        TBAAAccessInfo tbaaInfo, bool isInit,
                                        bool isNontemporal) {
  value = buildToMemory(value, ty);

  LValue atomicLValue =
      LValue::makeAddr(addr, ty, getContext(), baseInfo, tbaaInfo);
  if (ty->isAtomicType() ||
      (!isInit && LValueIsSuitableForInlineAtomic(atomicLValue))) {
    buildAtomicStore(RValue::get(value), atomicLValue, isInit);
    return;
  }

  mlir::Type srcTy = value.getType();
  if (const auto *clangVecTy = ty->getAs<clang::VectorType>()) {
    auto vecTy = dyn_cast<mlir::cir::VectorType>(srcTy);
    if (!CGM.getCodeGenOpts().PreserveVec3Type &&
        clangVecTy->getNumElements() == 3) {
      // Handle vec3 special.
      if (vecTy && vecTy.getSize() == 3) {
        // Our source is a vec3, do a shuffle vector to make it a vec4.
        value = builder.createVecShuffle(value.getLoc(), value,
                                         ArrayRef<int64_t>{0, 1, 2, -1});
        srcTy = mlir::cir::VectorType::get(vecTy.getContext(),
                                           vecTy.getEltType(), 4);
      }
      if (addr.getElementType() != srcTy) {
        addr = addr.withElementType(srcTy);
      }
    }
  }

  // Update the alloca with more info on initialization.
  assert(addr.getPointer() && "expected pointer to exist");
  auto srcAlloca =
      dyn_cast_or_null<mlir::cir::AllocaOp>(addr.getPointer().getDefiningOp());
  if (currVarDecl && srcAlloca) {
    const VarDecl *vd = currVarDecl;
    assert(vd && "VarDecl expected");
    if (vd->hasInit())
      srcAlloca.setInitAttr(mlir::UnitAttr::get(builder.getContext()));
  }

  assert(currSrcLoc && "must pass in source location");
  builder.createStore(*currSrcLoc, value, addr, isVolatile);

  if (isNontemporal) {
    llvm_unreachable("NYI");
  }

  if (MissingFeatures::tbaa())
    llvm_unreachable("NYI");
}

void CIRGenFunction::buildStoreOfScalar(mlir::Value value, LValue lvalue,
                                        bool isInit) {
  if (lvalue.getType()->isConstantMatrixType()) {
    llvm_unreachable("NYI");
  }

  buildStoreOfScalar(value, lvalue.getAddress(), lvalue.isVolatile(),
                     lvalue.getType(), lvalue.getBaseInfo(),
                     lvalue.getTBAAInfo(), isInit, lvalue.isNontemporal());
}

/// Given an expression that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result as an rvalue,
/// returning the rvalue.
RValue CIRGenFunction::buildLoadOfLValue(LValue lv, SourceLocation loc) {
  assert(!lv.getType()->isFunctionType());
  assert(!(lv.getType()->isConstantMatrixType()) && "not implemented");

  if (lv.isBitField())
    return buildLoadOfBitfieldLValue(lv, loc);

  if (lv.isSimple())
    return RValue::get(buildLoadOfScalar(lv, loc));

  if (lv.isVectorElt()) {
    auto load = builder.createLoad(getLoc(loc), lv.getVectorAddress());
    return RValue::get(builder.create<mlir::cir::VecExtractOp>(
        getLoc(loc), load, lv.getVectorIdx()));
  }

  if (lv.isExtVectorElt()) {
    return buildLoadOfExtVectorElementLValue(lv);
  }

  llvm_unreachable("NYI");
}

int64_t CIRGenFunction::getAccessedFieldNo(unsigned int idx,
                                           const mlir::ArrayAttr elts) {
  auto elt = mlir::dyn_cast<mlir::IntegerAttr>(elts[idx]);
  assert(elt && "The indices should be integer attributes");
  return elt.getInt();
}

// If this is a reference to a subset of the elements of a vector, create an
// appropriate shufflevector.
RValue CIRGenFunction::buildLoadOfExtVectorElementLValue(LValue lv) {
  mlir::Location loc = lv.getExtVectorPointer().getLoc();
  mlir::Value vec = builder.createLoad(loc, lv.getExtVectorAddress());

  // HLSL allows treating scalars as one-element vectors. Converting the scalar
  // IR value to a vector here allows the rest of codegen to behave as normal.
  if (getLangOpts().HLSL && !mlir::isa<mlir::cir::VectorType>(vec.getType())) {
    llvm_unreachable("HLSL NYI");
  }

  const mlir::ArrayAttr elts = lv.getExtVectorElts();

  // If the result of the expression is a non-vector type, we must be extracting
  // a single element.  Just codegen as an extractelement.
  const auto *exprVt = lv.getType()->getAs<clang::VectorType>();
  if (!exprVt) {
    int64_t inIdx = getAccessedFieldNo(0, elts);
    mlir::cir::ConstantOp elt =
        builder.getConstInt(loc, builder.getSInt64Ty(), inIdx);
    return RValue::get(builder.create<mlir::cir::VecExtractOp>(loc, vec, elt));
  }

  // Always use shuffle vector to try to retain the original program structure
  unsigned numResultElts = exprVt->getNumElements();

  SmallVector<int64_t, 4> mask;
  for (unsigned i = 0; i != numResultElts; ++i)
    mask.push_back(getAccessedFieldNo(i, elts));

  vec = builder.createVecShuffle(loc, vec, mask);
  return RValue::get(vec);
}

RValue CIRGenFunction::buildLoadOfBitfieldLValue(LValue lv,
                                                 SourceLocation loc) {
  const CIRGenBitFieldInfo &info = lv.getBitFieldInfo();

  // Get the output type.
  mlir::Type resLTy = convertType(lv.getType());
  Address ptr = lv.getBitFieldAddress();

  bool useVolatile = lv.isVolatileQualified() &&
                     info.VolatileStorageSize != 0 && isAAPCS(CGM.getTarget());

  auto field = builder.createGetBitfield(getLoc(loc), resLTy, ptr.getPointer(),
                                         ptr.getElementType(), info,
                                         lv.isVolatile(), useVolatile);
  assert(!MissingFeatures::emitScalarRangeCheck() && "NYI");
  return RValue::get(field);
}

void CIRGenFunction::buildStoreThroughExtVectorComponentLValue(RValue src,
                                                               LValue dst) {
  mlir::Location loc = dst.getExtVectorPointer().getLoc();

  // HLSL allows storing to scalar values through ExtVector component LValues.
  // To support this we need to handle the case where the destination address is
  // a scalar.
  Address dstAddr = dst.getExtVectorAddress();
  if (!mlir::isa<mlir::cir::VectorType>(dstAddr.getElementType())) {
    llvm_unreachable("HLSL NYI");
  }

  // This access turns into a read/modify/write of the vector.  Load the input
  // value now.
  mlir::Value vec = builder.createLoad(loc, dstAddr);
  const mlir::ArrayAttr elts = dst.getExtVectorElts();

  mlir::Value srcVal = src.getScalarVal();

  if (const clang::VectorType *vTy =
          dst.getType()->getAs<clang::VectorType>()) {
    unsigned numSrcElts = vTy->getNumElements();
    unsigned numDstElts = cast<mlir::cir::VectorType>(vec.getType()).getSize();
    if (numDstElts == numSrcElts) {
      // Use shuffle vector is the src and destination are the same number of
      // elements and restore the vector mask since it is on the side it will be
      // stored.
      SmallVector<int64_t, 4> mask(numDstElts);
      for (unsigned i = 0; i != numSrcElts; ++i)
        mask[getAccessedFieldNo(i, elts)] = i;

      vec = builder.createVecShuffle(loc, srcVal, mask);
    } else if (numDstElts > numSrcElts) {
      // Extended the source vector to the same length and then shuffle it
      // into the destination.
      // FIXME: since we're shuffling with undef, can we just use the indices
      //        into that?  This could be simpler.
      SmallVector<int64_t, 4> extMask;
      for (unsigned i = 0; i != numSrcElts; ++i)
        extMask.push_back(i);
      extMask.resize(numDstElts, -1);
      mlir::Value extSrcVal = builder.createVecShuffle(loc, srcVal, extMask);
      // build identity
      SmallVector<int64_t, 4> mask;
      for (unsigned i = 0; i != numDstElts; ++i)
        mask.push_back(i);

      // When the vector size is odd and .odd or .hi is used, the last element
      // of the Elts constant array will be one past the size of the vector.
      // Ignore the last element here, if it is greater than the mask size.
      if ((unsigned)getAccessedFieldNo(numSrcElts - 1, elts) == mask.size())
        numSrcElts--;

      // modify when what gets shuffled in
      for (unsigned i = 0; i != numSrcElts; ++i)
        mask[getAccessedFieldNo(i, elts)] = i + numDstElts;
      vec = builder.createVecShuffle(loc, vec, extSrcVal, mask);
    } else {
      // We should never shorten the vector
      llvm_unreachable("unexpected shorten vector length");
    }
  } else {
    // If the Src is a scalar (not a vector), and the target is a vector it must
    // be updating one element.
    unsigned inIdx = getAccessedFieldNo(0, elts);
    auto elt = builder.getSInt64(inIdx, loc);

    vec = builder.create<mlir::cir::VecInsertOp>(loc, vec, srcVal, elt);
  }

  builder.createStore(loc, vec, dst.getExtVectorAddress(),
                      dst.isVolatileQualified());
}

void CIRGenFunction::buildStoreThroughLValue(RValue src, LValue dst,
                                             bool isInit) {
  if (!dst.isSimple()) {
    if (dst.isVectorElt()) {
      // Read/modify/write the vector, inserting the new element
      mlir::Location loc = dst.getVectorPointer().getLoc();
      mlir::Value vector = builder.createLoad(loc, dst.getVectorAddress());
      vector = builder.create<mlir::cir::VecInsertOp>(
          loc, vector, src.getScalarVal(), dst.getVectorIdx());
      builder.createStore(loc, vector, dst.getVectorAddress());
      return;
    }

    if (dst.isExtVectorElt())
      return buildStoreThroughExtVectorComponentLValue(src, dst);

    assert(dst.isBitField() && "NIY LValue type");
    mlir::Value result;
    return buildStoreThroughBitfieldLValue(src, dst, result);
  }
  assert(dst.isSimple() && "only implemented simple");

  // There's special magic for assigning into an ARC-qualified l-value.
  if (Qualifiers::ObjCLifetime lifetime = dst.getQuals().getObjCLifetime()) {
    llvm_unreachable("NYI");
  }

  if (dst.isObjCWeak() && !dst.isNonGC()) {
    llvm_unreachable("NYI");
  }

  if (dst.isObjCStrong() && !dst.isNonGC()) {
    llvm_unreachable("NYI");
  }

  assert(src.isScalar() && "Can't emit an agg store with this method");
  buildStoreOfScalar(src.getScalarVal(), dst, isInit);
}

void CIRGenFunction::buildStoreThroughBitfieldLValue(RValue src, LValue dst,
                                                     mlir::Value &result) {
  // According to the AACPS:
  // When a volatile bit-field is written, and its container does not overlap
  // with any non-bit-field member, its container must be read exactly once
  // and written exactly once using the access width appropriate to the type
  // of the container. The two accesses are not atomic.
  if (dst.isVolatileQualified() && isAAPCS(CGM.getTarget()) &&
      CGM.getCodeGenOpts().ForceAAPCSBitfieldLoad)
    llvm_unreachable("volatile bit-field is not implemented for the AACPS");

  const CIRGenBitFieldInfo &info = dst.getBitFieldInfo();
  mlir::Type resLTy = getTypes().convertTypeForMem(dst.getType());
  Address ptr = dst.getBitFieldAddress();

  const bool useVolatile =
      CGM.getCodeGenOpts().AAPCSBitfieldWidth && dst.isVolatileQualified() &&
      info.VolatileStorageSize != 0 && isAAPCS(CGM.getTarget());

  mlir::Value dstAddr = dst.getAddress().getPointer();

  result = builder.createSetBitfield(
      dstAddr.getLoc(), resLTy, dstAddr, ptr.getElementType(),
      src.getScalarVal(), info, dst.isVolatileQualified(), useVolatile);
}

static LValue buildGlobalVarDeclLValue(CIRGenFunction &cgf, const Expr *e,
                                       const VarDecl *vd) {
  QualType t = e->getType();

  // If it's thread_local, emit a call to its wrapper function instead.
  if (vd->getTLSKind() == VarDecl::TLS_Dynamic &&
      cgf.CGM.getCXXABI().usesThreadWrapperFunction(vd))
    assert(0 && "not implemented");

  // Check if the variable is marked as declare target with link clause in
  // device codegen.
  if (cgf.getLangOpts().OpenMP)
    llvm_unreachable("not implemented");

  // Traditional LLVM codegen handles thread local separately, CIR handles
  // as part of getAddrOfGlobalVar.
  auto v = cgf.CGM.getAddrOfGlobalVar(vd);

  auto realVarTy = cgf.getTypes().convertTypeForMem(vd->getType());
  mlir::cir::PointerType realPtrTy = cgf.getBuilder().getPointerTo(
      realVarTy, cast_if_present<mlir::cir::AddressSpaceAttr>(
                     cast<mlir::cir::PointerType>(v.getType()).getAddrSpace()));
  if (realPtrTy != v.getType())
    v = cgf.getBuilder().createBitcast(v.getLoc(), v, realPtrTy);

  CharUnits alignment = cgf.getContext().getDeclAlign(vd);
  Address addr(v, realVarTy, alignment);
  // Emit reference to the private copy of the variable if it is an OpenMP
  // threadprivate variable.
  if (cgf.getLangOpts().OpenMP && !cgf.getLangOpts().OpenMPSimd &&
      vd->hasAttr<clang::OMPThreadPrivateDeclAttr>()) {
    assert(0 && "NYI");
  }
  LValue lv;
  if (vd->getType()->isReferenceType())
    assert(0 && "NYI");
  else
    lv = cgf.makeAddrLValue(addr, t, AlignmentSource::Decl);
  assert(!MissingFeatures::setObjCGCLValueClass() && "NYI");
  return lv;
}

static LValue buildCapturedFieldLValue(CIRGenFunction &cgf, const FieldDecl *fd,
                                       mlir::Value thisValue) {
  QualType tagType = cgf.getContext().getTagDeclType(fd->getParent());
  LValue lv = cgf.MakeNaturalAlignAddrLValue(thisValue, tagType);
  return cgf.buildLValueForField(lv, fd);
}

static LValue buildFunctionDeclLValue(CIRGenFunction &cgf, const Expr *e,
                                      GlobalDecl gd) {
  const FunctionDecl *fd = cast<FunctionDecl>(gd.getDecl());
  auto funcOp = buildFunctionDeclPointer(cgf.CGM, gd);
  auto loc = cgf.getLoc(e->getSourceRange());
  CharUnits align = cgf.getContext().getDeclAlign(fd);

  mlir::Type fnTy = funcOp.getFunctionType();
  auto ptrTy = mlir::cir::PointerType::get(cgf.getBuilder().getContext(), fnTy);
  mlir::Value addr = cgf.getBuilder().create<mlir::cir::GetGlobalOp>(
      loc, ptrTy, funcOp.getSymName());

  if (funcOp.getFunctionType() !=
      cgf.CGM.getTypes().ConvertType(fd->getType())) {
    fnTy = cgf.CGM.getTypes().ConvertType(fd->getType());
    ptrTy = mlir::cir::PointerType::get(cgf.getBuilder().getContext(), fnTy);

    addr = cgf.getBuilder().create<mlir::cir::CastOp>(
        addr.getLoc(), ptrTy, mlir::cir::CastKind::bitcast, addr);
  }

  return cgf.makeAddrLValue(Address(addr, fnTy, align), e->getType(),
                            AlignmentSource::Decl);
}

LValue CIRGenFunction::buildDeclRefLValue(const DeclRefExpr *e) {
  const NamedDecl *nd = e->getDecl();
  QualType t = e->getType();

  assert(e->isNonOdrUse() != NOUR_Unevaluated &&
         "should not emit an unevaluated operand");

  if (const auto *vd = dyn_cast<VarDecl>(nd)) {
    // Global Named registers access via intrinsics only
    if (vd->getStorageClass() == SC_Register && vd->hasAttr<AsmLabelAttr>() &&
        !vd->isLocalVarDecl())
      llvm_unreachable("NYI");

    assert(e->isNonOdrUse() != NOUR_Constant && "not implemented");

    // Check for captured variables.
    if (e->refersToEnclosingVariableOrCapture()) {
      vd = vd->getCanonicalDecl();
      if (auto *fd = LambdaCaptureFields.lookup(vd))
        return buildCapturedFieldLValue(*this, fd, CXXABIThisValue);
      assert(!MissingFeatures::CGCapturedStmtInfo() && "NYI");
      // TODO[OpenMP]: Find the appropiate captured variable value and return
      // it.
      // TODO[OpenMP]: Set non-temporal information in the captured LVal.
      // LLVM codegen:
      assert(!MissingFeatures::openMP());
      // Address addr = GetAddrOfBlockDecl(VD);
      // return MakeAddrLValue(addr, T, AlignmentSource::Decl);
    }
  }

  // FIXME(CIR): We should be able to assert this for FunctionDecls as well!
  // FIXME(CIR): We should be able to assert this for all DeclRefExprs, not just
  // those with a valid source location.
  assert((nd->isUsed(false) || !isa<VarDecl>(nd) || e->isNonOdrUse() ||
          !e->getLocation().isValid()) &&
         "Should not use decl without marking it used!");

  if (nd->hasAttr<WeakRefAttr>()) {
    llvm_unreachable("NYI");
  }

  if (const auto *vd = dyn_cast<VarDecl>(nd)) {
    // Check if this is a global variable
    if (vd->hasLinkage() || vd->isStaticDataMember())
      return buildGlobalVarDeclLValue(*this, e, vd);

    Address addr = Address::invalid();

    // The variable should generally be present in the local decl map.
    auto iter = LocalDeclMap.find(vd);
    if (iter != LocalDeclMap.end()) {
      addr = iter->second;
    }
    // Otherwise, it might be static local we haven't emitted yet for some
    // reason; most likely, because it's in an outer function.
    else if (vd->isStaticLocal()) {
      mlir::cir::GlobalOp var = CGM.getOrCreateStaticVarDecl(
          *vd, CGM.getCIRLinkageVarDefinition(vd, /*IsConstant=*/false));
      addr = Address(builder.createGetGlobal(var), convertType(vd->getType()),
                     getContext().getDeclAlign(vd));
    } else {
      llvm_unreachable("DeclRefExpr for decl not entered in LocalDeclMap?");
    }

    // Handle threadlocal function locals.
    if (vd->getTLSKind() != VarDecl::TLS_None)
      llvm_unreachable("thread-local storage is NYI");

    // Check for OpenMP threadprivate variables.
    if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd &&
        vd->hasAttr<OMPThreadPrivateDeclAttr>()) {
      llvm_unreachable("NYI");
    }

    // Drill into block byref variables.
    bool isBlockByref = vd->isEscapingByref();
    if (isBlockByref) {
      llvm_unreachable("NYI");
    }

    // Drill into reference types.
    LValue lv =
        vd->getType()->isReferenceType()
            ? buildLoadOfReferenceLValue(addr, getLoc(e->getSourceRange()),
                                         vd->getType(), AlignmentSource::Decl)
            : makeAddrLValue(addr, t, AlignmentSource::Decl);

    // Statics are defined as globals, so they are not include in the function's
    // symbol table.
    assert((vd->isStaticLocal() || symbolTable.count(vd)) &&
           "non-static locals should be already mapped");

    bool isLocalStorage = vd->hasLocalStorage();

    bool nonGCable =
        isLocalStorage && !vd->getType()->isReferenceType() && !isBlockByref;

    if (nonGCable && MissingFeatures::setNonGC()) {
      llvm_unreachable("garbage collection is NYI");
    }

    bool isImpreciseLifetime =
        (isLocalStorage && !vd->hasAttr<ObjCPreciseLifetimeAttr>());
    if (isImpreciseLifetime && MissingFeatures::ARC())
      llvm_unreachable("imprecise lifetime is NYI");
    assert(!MissingFeatures::setObjCGCLValueClass());

    // Statics are defined as globals, so they are not include in the function's
    // symbol table.
    assert((vd->isStaticLocal() || symbolTable.lookup(vd)) &&
           "Name lookup must succeed for non-static local variables");

    return lv;
  }

  if (const auto *fd = dyn_cast<FunctionDecl>(nd)) {
    LValue lv = buildFunctionDeclLValue(*this, e, fd);

    // Emit debuginfo for the function declaration if the target wants to.
    if (getContext().getTargetInfo().allowDebugInfoForExternalRef())
      assert(!MissingFeatures::generateDebugInfo());

    return lv;
  }

  // FIXME: While we're emitting a binding from an enclosing scope, all other
  // DeclRefExprs we see should be implicitly treated as if they also refer to
  // an enclosing scope.
  if (const auto *bd = dyn_cast<BindingDecl>(nd)) {
    if (e->refersToEnclosingVariableOrCapture()) {
      auto *fd = LambdaCaptureFields.lookup(bd);
      return buildCapturedFieldLValue(*this, fd, CXXABIThisValue);
    }
    return buildLValue(bd->getBinding());
  }

  // We can form DeclRefExprs naming GUID declarations when reconstituting
  // non-type template parameters into expressions.
  if (const auto *gd = dyn_cast<MSGuidDecl>(nd))
    llvm_unreachable("NYI");

  if (const auto *tpo = dyn_cast<TemplateParamObjectDecl>(nd))
    llvm_unreachable("NYI");

  llvm_unreachable("Unhandled DeclRefExpr");
}

LValue
CIRGenFunction::buildPointerToDataMemberBinaryExpr(const BinaryOperator *e) {
  assert((e->getOpcode() == BO_PtrMemD || e->getOpcode() == BO_PtrMemI) &&
         "unexpected binary operator opcode");

  auto baseAddr = Address::invalid();
  if (e->getOpcode() == BO_PtrMemD)
    baseAddr = buildLValue(e->getLHS()).getAddress();
  else
    baseAddr = buildPointerWithAlignment(e->getLHS());

  const auto *memberPtrTy = e->getRHS()->getType()->castAs<MemberPointerType>();

  auto memberPtr = buildScalarExpr(e->getRHS());

  LValueBaseInfo baseInfo;
  // TODO(cir): add TBAA
  assert(!MissingFeatures::tbaa());
  auto memberAddr = buildCXXMemberDataPointerAddress(e, baseAddr, memberPtr,
                                                     memberPtrTy, &baseInfo);

  return makeAddrLValue(memberAddr, memberPtrTy->getPointeeType(), baseInfo);
}

LValue
CIRGenFunction::buildExtVectorElementExpr(const ExtVectorElementExpr *e) {
  // Emit the base vector as an l-value.
  LValue base;

  // ExtVectorElementExpr's base can either be a vector or pointer to vector.
  if (e->isArrow()) {
    // If it is a pointer to a vector, emit the address and form an lvalue with
    // it.
    LValueBaseInfo baseInfo;
    // TODO(cir): Support TBAA
    assert(!MissingFeatures::tbaa());
    Address ptr = buildPointerWithAlignment(e->getBase(), &baseInfo);
    const auto *pt = e->getBase()->getType()->castAs<clang::PointerType>();
    base = makeAddrLValue(ptr, pt->getPointeeType(), baseInfo);
    base.getQuals().removeObjCGCAttr();
  } else if (e->getBase()->isGLValue()) {
    // Otherwise, if the base is an lvalue ( as in the case of foo.x.x),
    // emit the base as an lvalue.
    assert(e->getBase()->getType()->isVectorType());
    base = buildLValue(e->getBase());
  } else {
    // Otherwise, the base is a normal rvalue (as in (V+V).x), emit it as such.
    assert(e->getBase()->getType()->isVectorType() &&
           "Result must be a vector");
    mlir::Value vec = buildScalarExpr(e->getBase());

    // Store the vector to memory (because LValue wants an address).
    QualType baseTy = e->getBase()->getType();
    Address vecMem = CreateMemTemp(baseTy, vec.getLoc(), "tmp");
    builder.createStore(vec.getLoc(), vec, vecMem);
    base = makeAddrLValue(vecMem, baseTy, AlignmentSource::Decl);
  }

  QualType type =
      e->getType().withCVRQualifiers(base.getQuals().getCVRQualifiers());

  // Encode the element access list into a vector of unsigned indices.
  SmallVector<uint32_t, 4> indices;
  e->getEncodedElementAccess(indices);

  if (base.isSimple()) {
    SmallVector<int64_t, 4> attrElts;
    for (uint32_t i : indices) {
      attrElts.push_back(static_cast<int64_t>(i));
    }
    auto elts = builder.getI64ArrayAttr(attrElts);
    return LValue::MakeExtVectorElt(base.getAddress(), elts, type,
                                    base.getBaseInfo(), base.getTBAAInfo());
  }
  assert(base.isExtVectorElt() && "Can only subscript lvalue vec elts here!");

  mlir::ArrayAttr baseElts = base.getExtVectorElts();

  // Composite the two indices
  SmallVector<int64_t, 4> attrElts;
  for (uint32_t i : indices) {
    attrElts.push_back(getAccessedFieldNo(i, baseElts));
  }
  auto elts = builder.getI64ArrayAttr(attrElts);

  return LValue::MakeExtVectorElt(base.getExtVectorAddress(), elts, type,
                                  base.getBaseInfo(), base.getTBAAInfo());
}

LValue CIRGenFunction::buildBinaryOperatorLValue(const BinaryOperator *e) {
  // Comma expressions just emit their LHS then their RHS as an l-value.
  if (e->getOpcode() == BO_Comma) {
    buildIgnoredExpr(e->getLHS());
    return buildLValue(e->getRHS());
  }

  if (e->getOpcode() == BO_PtrMemD || e->getOpcode() == BO_PtrMemI)
    return buildPointerToDataMemberBinaryExpr(e);

  assert(e->getOpcode() == BO_Assign && "unexpected binary l-value");

  // Note that in all of these cases, __block variables need the RHS
  // evaluated first just in case the variable gets moved by the RHS.

  switch (CIRGenFunction::getEvaluationKind(e->getType())) {
  case TEK_Scalar: {
    assert(e->getLHS()->getType().getObjCLifetime() ==
               clang::Qualifiers::ObjCLifetime::OCL_None &&
           "not implemented");

    RValue rv = buildAnyExpr(e->getRHS());
    LValue lv = buildLValue(e->getLHS());

    SourceLocRAIIObject loc{*this, getLoc(e->getSourceRange())};
    if (lv.isBitField()) {
      mlir::Value result;
      buildStoreThroughBitfieldLValue(rv, lv, result);
    } else {
      buildStoreThroughLValue(rv, lv);
    }
    if (getLangOpts().OpenMP)
      CGM.getOpenMPRuntime().checkAndEmitLastprivateConditional(*this,
                                                                e->getLHS());
    return lv;
  }

  case TEK_Complex:
    return buildComplexAssignmentLValue(e);
  case TEK_Aggregate:
    assert(0 && "not implemented");
  }
  llvm_unreachable("bad evaluation kind");
}

/// Given an expression of pointer type, try to
/// derive a more accurate bound on the alignment of the pointer.
Address CIRGenFunction::buildPointerWithAlignment(
    const Expr *expr, LValueBaseInfo *baseInfo, TBAAAccessInfo *tbaaInfo,
    KnownNonNull_t isKnownNonNull) {
  Address addr = ::buildPointerWithAlignment(expr, baseInfo, tbaaInfo,
                                             isKnownNonNull, *this);
  if (isKnownNonNull && !addr.isKnownNonNull())
    addr.setKnownNonNull();
  return addr;
}

/// Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
mlir::Value CIRGenFunction::evaluateExprAsBool(const Expr *e) {
  // TODO(cir): PGO
  if (const MemberPointerType *mpt = e->getType()->getAs<MemberPointerType>()) {
    assert(0 && "not implemented");
  }

  QualType boolTy = getContext().BoolTy;
  SourceLocation loc = e->getExprLoc();
  // TODO(cir): CGFPOptionsRAII for FP stuff.
  if (!e->getType()->isAnyComplexType())
    return buildScalarConversion(buildScalarExpr(e), e->getType(), boolTy, loc);

  llvm_unreachable("complex to scalar not implemented");
}

LValue CIRGenFunction::buildUnaryOpLValue(const UnaryOperator *e) {
  // __extension__ doesn't affect lvalue-ness.
  if (e->getOpcode() == UO_Extension)
    return buildLValue(e->getSubExpr());

  QualType exprTy = getContext().getCanonicalType(e->getSubExpr()->getType());
  switch (e->getOpcode()) {
  default:
    llvm_unreachable("Unknown unary operator lvalue!");
  case UO_Deref: {
    QualType t = e->getSubExpr()->getType()->getPointeeType();
    assert(!t.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");

    LValueBaseInfo baseInfo;
    // TODO: add TBAAInfo
    Address addr = buildPointerWithAlignment(e->getSubExpr(), &baseInfo);

    // Tag 'load' with deref attribute.
    if (auto loadOp =
            dyn_cast<::mlir::cir::LoadOp>(addr.getPointer().getDefiningOp())) {
      loadOp.setIsDerefAttr(mlir::UnitAttr::get(builder.getContext()));
    }

    LValue lv = LValue::makeAddr(addr, t, baseInfo);
    // TODO: set addr space
    // TODO: ObjC/GC/__weak write barrier stuff.
    return lv;
  }
  case UO_Real:
  case UO_Imag: {
    LValue lv = buildLValue(e->getSubExpr());
    assert(lv.isSimple() && "real/imag on non-ordinary l-value");

    // __real is valid on scalars.  This is a faster way of testing that.
    // __imag can only produce an rvalue on scalars.
    if (e->getOpcode() == UO_Real &&
        !mlir::isa<mlir::cir::ComplexType>(lv.getAddress().getElementType())) {
      assert(e->getSubExpr()->getType()->isArithmeticType());
      return lv;
    }

    QualType t = exprTy->castAs<clang::ComplexType>()->getElementType();

    auto loc = getLoc(e->getExprLoc());
    Address component =
        (e->getOpcode() == UO_Real
             ? buildAddrOfRealComponent(loc, lv.getAddress(), lv.getType())
             : buildAddrOfImagComponent(loc, lv.getAddress(), lv.getType()));
    // TODO(cir): TBAA info.
    assert(!MissingFeatures::tbaa());
    LValue elemLv = makeAddrLValue(component, t, lv.getBaseInfo());
    elemLv.getQuals().addQualifiers(lv.getQuals());
    return elemLv;
  }
  case UO_PreInc:
  case UO_PreDec: {
    bool isInc = e->isIncrementOp();
    bool isPre = e->isPrefix();
    LValue lv = buildLValue(e->getSubExpr());

    if (e->getType()->isAnyComplexType()) {
      buildComplexPrePostIncDec(e, lv, isInc, true /*isPre*/);
    } else {
      buildScalarPrePostIncDec(e, lv, isInc, isPre);
    }

    return lv;
  }
  }
}

/// Emit code to compute the specified expression which
/// can have any type.  The result is returned as an RValue struct.
RValue CIRGenFunction::buildAnyExpr(const Expr *e, AggValueSlot aggSlot,
                                    bool ignoreResult) {
  switch (CIRGenFunction::getEvaluationKind(e->getType())) {
  case TEK_Scalar:
    return RValue::get(buildScalarExpr(e));
  case TEK_Complex:
    return RValue::getComplex(buildComplexExpr(e));
  case TEK_Aggregate: {
    if (!ignoreResult && aggSlot.isIgnored())
      aggSlot = CreateAggTemp(e->getType(), getLoc(e->getSourceRange()),
                              getCounterAggTmpAsString());
    buildAggExpr(e, aggSlot);
    return aggSlot.asRValue();
  }
  }
  llvm_unreachable("bad evaluation kind");
}

RValue CIRGenFunction::buildCallExpr(const clang::CallExpr *e,
                                     ReturnValueSlot returnValue) {
  assert(!e->getCallee()->getType()->isBlockPointerType() && "ObjC Blocks NYI");

  if (const auto *ce = dyn_cast<CXXMemberCallExpr>(e))
    return buildCXXMemberCallExpr(ce, returnValue);

  assert(!dyn_cast<CUDAKernelCallExpr>(e) && "CUDA NYI");
  if (const auto *ce = dyn_cast<CXXOperatorCallExpr>(e))
    if (const CXXMethodDecl *md =
            dyn_cast_or_null<CXXMethodDecl>(ce->getCalleeDecl()))
      return buildCXXOperatorMemberCallExpr(ce, md, returnValue);

  CIRGenCallee callee = buildCallee(e->getCallee());

  if (callee.isBuiltin())
    return buildBuiltinExpr(callee.getBuiltinDecl(), callee.getBuiltinID(), e,
                            returnValue);

  assert(!callee.isPsuedoDestructor() && "NYI");

  return buildCall(e->getCallee()->getType(), callee, e, returnValue);
}

RValue CIRGenFunction::GetUndefRValue(QualType ty) {
  if (ty->isVoidType())
    return RValue::get(nullptr);

  switch (getEvaluationKind(ty)) {
  case TEK_Complex: {
    llvm_unreachable("NYI");
  }

  // If this is a use of an undefined aggregate type, the aggregate must have
  // an identifiable address. Just because the contents of the value are
  // undefined doesn't mean that the address can't be taken and compared.
  case TEK_Aggregate: {
    llvm_unreachable("NYI");
  }

  case TEK_Scalar:
    llvm_unreachable("NYI");
  }
  llvm_unreachable("bad evaluation kind");
}

LValue CIRGenFunction::buildStmtExprLValue(const StmtExpr *e) {
  // Can only get l-value for message expression returning aggregate type
  RValue rv = buildAnyExprToTemp(e);
  return makeAddrLValue(rv.getAggregateAddress(), e->getType(),
                        AlignmentSource::Decl);
}

RValue CIRGenFunction::buildCall(clang::QualType calleeType,
                                 const CIRGenCallee &origCallee,
                                 const clang::CallExpr *e,
                                 ReturnValueSlot returnValue,
                                 mlir::Value chain) {
  // Get the actual function type. The callee type will always be a pointer to
  // function type or a block pointer type.
  assert(calleeType->isFunctionPointerType() &&
         "Call must have function pointer type!");

  auto *targetDecl = origCallee.getAbstractInfo().getCalleeDecl().getDecl();
  (void)targetDecl;

  calleeType = getContext().getCanonicalType(calleeType);

  auto pointeeType = cast<clang::PointerType>(calleeType)->getPointeeType();

  CIRGenCallee callee = origCallee;

  if (getLangOpts().CPlusPlus)
    assert(!SanOpts.has(SanitizerKind::Function) && "Sanitizers NYI");

  const auto *fnType = cast<FunctionType>(pointeeType);

  assert(!SanOpts.has(SanitizerKind::CFIICall) && "Sanitizers NYI");

  CallArgList args;

  assert(!chain && "FIX THIS");

  // C++17 requires that we evaluate arguments to a call using assignment syntax
  // right-to-left, and that we evaluate arguments to certain other operators
  // left-to-right. Note that we allow this to override the order dictated by
  // the calling convention on the MS ABI, which means that parameter
  // destruction order is not necessarily reverse construction order.
  // FIXME: Revisit this based on C++ committee response to unimplementability.
  EvaluationOrder order = EvaluationOrder::Default;
  if (auto *oce = dyn_cast<CXXOperatorCallExpr>(e)) {
    if (oce->isAssignmentOp())
      order = EvaluationOrder::ForceRightToLeft;
    else {
      switch (oce->getOperator()) {
      case OO_LessLess:
      case OO_GreaterGreater:
      case OO_AmpAmp:
      case OO_PipePipe:
      case OO_Comma:
      case OO_ArrowStar:
        order = EvaluationOrder::ForceLeftToRight;
        break;
      default:
        break;
      }
    }
  }

  buildCallArgs(args, dyn_cast<FunctionProtoType>(fnType), e->arguments(),
                e->getDirectCallee(), /*ParamsToSkip*/ 0, order);

  const CIRGenFunctionInfo &fnInfo = CGM.getTypes().arrangeFreeFunctionCall(
      args, fnType, /*ChainCall=*/chain.getAsOpaquePointer());

  // C99 6.5.2.2p6:
  //   If the expression that denotes the called function has a type that does
  //   not include a prototype, [the default argument promotions are performed].
  //   If the number of arguments does not equal the number of parameters, the
  //   behavior is undefined. If the function is defined with at type that
  //   includes a prototype, and either the prototype ends with an ellipsis (,
  //   ...) or the types of the arguments after promotion are not compatible
  //   with the types of the parameters, the behavior is undefined. If the
  //   function is defined with a type that does not include a prototype, and
  //   the types of the arguments after promotion are not compatible with those
  //   of the parameters after promotion, the behavior is undefined [except in
  //   some trivial cases].
  // That is, in the general case, we should assume that a call through an
  // unprototyped function type works like a *non-variadic* call. The way we
  // make this work is to cast to the exxact type fo the promoted arguments.
  //
  // Chain calls use the same code path to add the inviisble chain parameter to
  // the function type.
  if (isa<FunctionNoProtoType>(fnType) || chain) {
    assert(!MissingFeatures::chainCall());
    assert(!MissingFeatures::addressSpace());
    auto calleeTy = getTypes().GetFunctionType(fnInfo);
    // get non-variadic function type
    calleeTy = mlir::cir::FuncType::get(calleeTy.getInputs(),
                                        calleeTy.getReturnType(), false);
    auto calleePtrTy =
        mlir::cir::PointerType::get(builder.getContext(), calleeTy);

    auto *fn = callee.getFunctionPointer();
    mlir::Value addr;
    if (auto funcOp = llvm::dyn_cast<mlir::cir::FuncOp>(fn)) {
      addr = builder.create<mlir::cir::GetGlobalOp>(
          getLoc(e->getSourceRange()),
          mlir::cir::PointerType::get(builder.getContext(),
                                      funcOp.getFunctionType()),
          funcOp.getSymName());
    } else {
      addr = fn->getResult(0);
    }

    fn = builder.createBitcast(addr, calleePtrTy).getDefiningOp();
    callee.setFunctionPointer(fn);
  }

  assert(!CGM.getLangOpts().HIP && "HIP NYI");

  assert(!MustTailCall && "Must tail NYI");
  mlir::cir::CIRCallOpInterface callOP;
  RValue call = buildCall(fnInfo, callee, returnValue, args, &callOP,
                          e == MustTailCall, getLoc(e->getExprLoc()), e);

  assert(!getDebugInfo() && "Debug Info NYI");

  return call;
}

/// Emit code to compute the specified expression, ignoring the result.
void CIRGenFunction::buildIgnoredExpr(const Expr *e) {
  if (e->isPRValue())
    return (void)buildAnyExpr(e, AggValueSlot::ignored(), true);

  // Just emit it as an l-value and drop the result.
  buildLValue(e);
}

Address CIRGenFunction::buildArrayToPointerDecay(const Expr *e,
                                                 LValueBaseInfo *baseInfo) {
  assert(e->getType()->isArrayType() &&
         "Array to pointer decay must have array source type!");

  // Expressions of array type can't be bitfields or vector elements.
  LValue lv = buildLValue(e);
  Address addr = lv.getAddress();

  // If the array type was an incomplete type, we need to make sure
  // the decay ends up being the right type.
  auto lvalueAddrTy =
      mlir::dyn_cast<mlir::cir::PointerType>(addr.getPointer().getType());
  assert(lvalueAddrTy && "expected pointer");

  if (e->getType()->isVariableArrayType())
    return addr;

  auto pointeeTy =
      mlir::dyn_cast<mlir::cir::ArrayType>(lvalueAddrTy.getPointee());
  assert(pointeeTy && "expected array");

  mlir::Type arrayTy = convertType(e->getType());
  assert(mlir::isa<mlir::cir::ArrayType>(arrayTy) && "expected array");
  assert(pointeeTy == arrayTy);

  // The result of this decay conversion points to an array element within the
  // base lvalue. However, since TBAA currently does not support representing
  // accesses to elements of member arrays, we conservatively represent accesses
  // to the pointee object as if it had no any base lvalue specified.
  // TODO: Support TBAA for member arrays.
  QualType eltType = e->getType()->castAsArrayTypeUnsafe()->getElementType();
  if (baseInfo)
    *baseInfo = lv.getBaseInfo();
  assert(!MissingFeatures::tbaa() && "NYI");

  mlir::Value ptr = CGM.getBuilder().maybeBuildArrayDecay(
      CGM.getLoc(e->getSourceRange()), addr.getPointer(),
      getTypes().convertTypeForMem(eltType));
  return Address(ptr, addr.getAlignment());
}

/// If the specified expr is a simple decay from an array to pointer,
/// return the array subexpression.
/// FIXME: this could be abstracted into a commeon AST helper.
static const Expr *isSimpleArrayDecayOperand(const Expr *e) {
  // If this isn't just an array->pointer decay, bail out.
  const auto *ce = dyn_cast<CastExpr>(e);
  if (!ce || ce->getCastKind() != CK_ArrayToPointerDecay)
    return nullptr;

  // If this is a decay from variable width array, bail out.
  const Expr *subExpr = ce->getSubExpr();
  if (subExpr->getType()->isVariableArrayType())
    return nullptr;

  return subExpr;
}

/// Given an array base, check whether its member access belongs to a record
/// with preserve_access_index attribute or not.
/// TODO(cir): don't need to be specific to LLVM's codegen, refactor into common
/// AST helpers.
static bool isPreserveAIArrayBase(CIRGenFunction &cgf, const Expr *arrayBase) {
  if (!arrayBase || !cgf.getDebugInfo())
    return false;

  // Only support base as either a MemberExpr or DeclRefExpr.
  // DeclRefExpr to cover cases like:
  //    struct s { int a; int b[10]; };
  //    struct s *p;
  //    p[1].a
  // p[1] will generate a DeclRefExpr and p[1].a is a MemberExpr.
  // p->b[5] is a MemberExpr example.
  const Expr *e = arrayBase->IgnoreImpCasts();
  if (const auto *me = dyn_cast<MemberExpr>(e))
    return me->getMemberDecl()->hasAttr<BPFPreserveAccessIndexAttr>();

  if (const auto *dre = dyn_cast<DeclRefExpr>(e)) {
    const auto *varDef = dyn_cast<VarDecl>(dre->getDecl());
    if (!varDef)
      return false;

    const auto *ptrT = varDef->getType()->getAs<clang::PointerType>();
    if (!ptrT)
      return false;

    const auto *pointeeT =
        ptrT->getPointeeType()->getUnqualifiedDesugaredType();
    if (const auto *recT = dyn_cast<RecordType>(pointeeT))
      return recT->getDecl()->hasAttr<BPFPreserveAccessIndexAttr>();
    return false;
  }

  return false;
}

static mlir::IntegerAttr getConstantIndexOrNull(mlir::Value idx) {
  // TODO(cir): should we consider using MLIRs IndexType instead of IntegerAttr?
  if (auto constantOp = dyn_cast<mlir::cir::ConstantOp>(idx.getDefiningOp()))
    return mlir::dyn_cast<mlir::IntegerAttr>(constantOp.getValue());
  return {};
}

static CharUnits getArrayElementAlign(CharUnits arrayAlign, mlir::Value idx,
                                      CharUnits eltSize) {
  // If we have a constant index, we can use the exact offset of the
  // element we're accessing.
  auto constantIdx = getConstantIndexOrNull(idx);
  if (constantIdx) {
    CharUnits offset = constantIdx.getValue().getZExtValue() * eltSize;
    return arrayAlign.alignmentAtOffset(offset);
    // Otherwise, use the worst-case alignment for any element.
  }
  return arrayAlign.alignmentOfArrayElement(eltSize);
}

static mlir::Value
buildArraySubscriptPtr(CIRGenFunction &cgf, mlir::Location beginLoc,
                       mlir::Location endLoc, mlir::Value ptr, mlir::Type eltTy,
                       ArrayRef<mlir::Value> indices, bool inbounds,
                       bool signedIndices, bool shouldDecay,
                       const llvm::Twine &name = "arrayidx") {
  assert(indices.size() == 1 && "cannot handle multiple indices yet");
  auto idx = indices.back();
  auto &cgm = cgf.getCIRGenModule();
  // TODO(cir): LLVM codegen emits in bound gep check here, is there anything
  // that would enhance tracking this later in CIR?
  if (inbounds)
    assert(!MissingFeatures::emitCheckedInBoundsGEP() && "NYI");
  return cgm.getBuilder().getArrayElement(beginLoc, endLoc, ptr, eltTy, idx,
                                          shouldDecay);
}

static QualType getFixedSizeElementType(const ASTContext &ctx,
                                        const VariableArrayType *vla) {
  QualType eltType;
  do {
    eltType = vla->getElementType();
  } while ((vla = ctx.getAsVariableArrayType(eltType)));
  return eltType;
}

static Address buildArraySubscriptPtr(
    CIRGenFunction &cgf, mlir::Location beginLoc, mlir::Location endLoc,
    Address addr, ArrayRef<mlir::Value> indices, QualType eltType,
    bool inbounds, bool signedIndices, mlir::Location loc, bool shouldDecay,
    QualType *arrayType = nullptr, const Expr *base = nullptr,
    const llvm::Twine &name = "arrayidx") {
  // Determine the element size of the statically-sized base.  This is
  // the thing that the indices are expressed in terms of.
  if (const auto *vla = cgf.getContext().getAsVariableArrayType(eltType)) {
    eltType = getFixedSizeElementType(cgf.getContext(), vla);
  }

  // We can use that to compute the best alignment of the element.
  CharUnits eltSize = cgf.getContext().getTypeSizeInChars(eltType);
  CharUnits eltAlign =
      getArrayElementAlign(addr.getAlignment(), indices.back(), eltSize);

  mlir::Value eltPtr;
  auto lastIndex = getConstantIndexOrNull(indices.back());
  if (!lastIndex ||
      (!cgf.IsInPreservedAIRegion && !isPreserveAIArrayBase(cgf, base))) {
    eltPtr = buildArraySubscriptPtr(cgf, beginLoc, endLoc, addr.getPointer(),
                                    addr.getElementType(), indices, inbounds,
                                    signedIndices, shouldDecay, name);
  } else {
    // assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
    // assert(indices.size() == 1 && "cannot handle multiple indices yet");
    // auto idx = indices.back();
    // auto &CGM = CGF.getCIRGenModule();
    // eltPtr = CGM.getBuilder().getArrayElement(beginLoc, endLoc,
    //                             addr.getPointer(), addr.getElementType(),
    //                             idx);
    assert(0 && "NYI");
  }

  return Address(eltPtr, cgf.getTypes().convertTypeForMem(eltType), eltAlign);
}

LValue CIRGenFunction::buildArraySubscriptExpr(const ArraySubscriptExpr *e,
                                               bool accessed) {
  // The index must always be an integer, which is not an aggregate.  Emit it
  // in lexical order (this complexity is, sadly, required by C++17).
  mlir::Value idxPre =
      (e->getLHS() == e->getIdx()) ? buildScalarExpr(e->getIdx()) : nullptr;
  bool signedIndices = false;
  auto emitIdxAfterBase = [&, idxPre](bool promote) -> mlir::Value {
    mlir::Value idx = idxPre;
    if (e->getLHS() != e->getIdx()) {
      assert(e->getRHS() == e->getIdx() && "index was neither LHS nor RHS");
      idx = buildScalarExpr(e->getIdx());
    }

    QualType idxTy = e->getIdx()->getType();
    bool idxSigned = idxTy->isSignedIntegerOrEnumerationType();
    signedIndices |= idxSigned;

    if (SanOpts.has(SanitizerKind::ArrayBounds))
      llvm_unreachable("array bounds sanitizer is NYI");

    // Extend or truncate the index type to 32 or 64-bits.
    auto ptrTy = mlir::dyn_cast<mlir::cir::PointerType>(idx.getType());
    if (promote && ptrTy && mlir::isa<mlir::cir::IntType>(ptrTy.getPointee()))
      llvm_unreachable("index type cast is NYI");

    return idx;
  };
  idxPre = nullptr;

  // If the base is a vector type, then we are forming a vector element
  // with this subscript.
  if (e->getBase()->getType()->isVectorType() &&
      !isa<ExtVectorElementExpr>(e->getBase())) {
    LValue lhs = buildLValue(e->getBase());
    auto index = emitIdxAfterBase(/*Promote=*/false);
    return LValue::MakeVectorElt(lhs.getAddress(), index,
                                 e->getBase()->getType(), lhs.getBaseInfo(),
                                 lhs.getTBAAInfo());
  }

  // All the other cases basically behave like simple offsetting.

  // Handle the extvector case we ignored above.
  if (isa<ExtVectorElementExpr>(e->getBase())) {
    llvm_unreachable("extvector subscript is NYI");
  }

  assert(!MissingFeatures::tbaa() && "TBAA is NYI");
  LValueBaseInfo eltBaseInfo;
  Address addr = Address::invalid();
  if (const VariableArrayType *vla =
          getContext().getAsVariableArrayType(e->getType())) {
    // The base must be a pointer, which is not an aggregate.  Emit
    // it.  It needs to be emitted first in case it's what captures
    // the VLA bounds.
    addr = buildPointerWithAlignment(e->getBase(), &eltBaseInfo);
    auto idx = emitIdxAfterBase(/*Promote*/ true);

    // The element count here is the total number of non-VLA elements.
    mlir::Value numElements = getVLASize(vla).NumElts;
    idx = builder.createCast(mlir::cir::CastKind::integral, idx,
                             numElements.getType());
    idx = builder.createMul(idx, numElements);

    QualType ptrType = e->getBase()->getType();
    addr = buildArraySubscriptPtr(
        *this, CGM.getLoc(e->getBeginLoc()), CGM.getLoc(e->getEndLoc()), addr,
        {idx}, e->getType(), !getLangOpts().isSignedOverflowDefined(),
        signedIndices, CGM.getLoc(e->getExprLoc()), /*shouldDecay=*/false,
        &ptrType, e->getBase());
  } else if (const ObjCObjectType *oit =
                 e->getType()->getAs<ObjCObjectType>()) {
    llvm_unreachable("ObjC object type subscript is NYI");
  } else if (const Expr *array = isSimpleArrayDecayOperand(e->getBase())) {
    // If this is A[i] where A is an array, the frontend will have decayed
    // the base to be a ArrayToPointerDecay implicit cast.  While correct, it is
    // inefficient at -O0 to emit a "gep A, 0, 0" when codegen'ing it, then
    // a "gep x, i" here.  Emit one "gep A, 0, i".
    assert(array->getType()->isArrayType() &&
           "Array to pointer decay must have array source type!");
    LValue arrayLv;
    // For simple multidimensional array indexing, set the 'accessed' flag
    // for better bounds-checking of the base expression.
    if (const auto *ase = dyn_cast<ArraySubscriptExpr>(array))
      arrayLv = buildArraySubscriptExpr(ase, /*Accessed=*/true);
    else
      arrayLv = buildLValue(array);
    auto idx = emitIdxAfterBase(/*Promote=*/true);

    // Propagate the alignment from the array itself to the result.
    QualType arrayType = array->getType();
    addr = buildArraySubscriptPtr(
        *this, CGM.getLoc(array->getBeginLoc()), CGM.getLoc(array->getEndLoc()),
        arrayLv.getAddress(), {idx}, e->getType(),
        !getLangOpts().isSignedOverflowDefined(), signedIndices,
        CGM.getLoc(e->getExprLoc()), /*shouldDecay=*/true, &arrayType,
        e->getBase());
    eltBaseInfo = arrayLv.getBaseInfo();
    // TODO(cir): EltTBAAInfo
    assert(!MissingFeatures::tbaa() && "TBAA is NYI");
  } else {
    // The base must be a pointer; emit it with an estimate of its alignment.
    // TODO(cir): EltTBAAInfo
    assert(!MissingFeatures::tbaa() && "TBAA is NYI");
    addr = buildPointerWithAlignment(e->getBase(), &eltBaseInfo);
    auto idx = emitIdxAfterBase(/*Promote*/ true);
    QualType ptrType = e->getBase()->getType();
    addr = buildArraySubscriptPtr(
        *this, CGM.getLoc(e->getBeginLoc()), CGM.getLoc(e->getEndLoc()), addr,
        idx, e->getType(), !getLangOpts().isSignedOverflowDefined(),
        signedIndices, CGM.getLoc(e->getExprLoc()), /*shouldDecay=*/false,
        &ptrType, e->getBase());
  }

  LValue lv = LValue::makeAddr(addr, e->getType(), eltBaseInfo);

  if (getLangOpts().ObjC && getLangOpts().getGC() != LangOptions::NonGC) {
    llvm_unreachable("ObjC is NYI");
  }

  return lv;
}

LValue CIRGenFunction::buildStringLiteralLValue(const StringLiteral *e) {
  auto sym = CGM.getAddrOfConstantStringFromLiteral(e).getSymbol();

  auto *cstGlobal = mlir::SymbolTable::lookupSymbolIn(CGM.getModule(), sym);
  assert(cstGlobal && "Expected global");

  auto g = dyn_cast<mlir::cir::GlobalOp>(cstGlobal);
  assert(g && "unaware of other symbol providers");

  auto ptrTy = mlir::cir::PointerType::get(CGM.getBuilder().getContext(),
                                           g.getSymType());
  assert(g.getAlignment() && "expected alignment for string literal");
  auto align = *g.getAlignment();
  auto addr = builder.create<mlir::cir::GetGlobalOp>(
      getLoc(e->getSourceRange()), ptrTy, g.getSymName());
  return makeAddrLValue(
      Address(addr, g.getSymType(), CharUnits::fromQuantity(align)),
      e->getType(), AlignmentSource::Decl);
}

/// Casts are never lvalues unless that cast is to a reference type. If the cast
/// is to a reference, we can have the usual lvalue result, otherwise if a cast
/// is needed by the code generator in an lvalue context, then it must mean that
/// we need the address of an aggregate in order to access one of its members.
/// This can happen for all the reasons that casts are permitted with aggregate
/// result, including noop aggregate casts, and cast from scalar to union.
LValue CIRGenFunction::buildCastLValue(const CastExpr *e) {
  switch (e->getCastKind()) {
  case CK_HLSLArrayRValue:
  case CK_HLSLVectorTruncation:
  case CK_ToVoid:
  case CK_BitCast:
  case CK_LValueToRValueBitCast:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToMemberPointer:
  case CK_NullToPointer:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_BooleanToSignedIntegral:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_DerivedToBaseMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_IntToOCLSampler:
  case CK_FloatingToFixedPoint:
  case CK_FixedPointToFloating:
  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToIntegral:
  case CK_IntegralToFixedPoint:
  case CK_MatrixCast:
    llvm_unreachable("NYI");

  case CK_Dependent:
    llvm_unreachable("dependent cast kind in IR gen!");

  case CK_BuiltinFnToFnPtr:
    llvm_unreachable("builtin functions are handled elsewhere");

  // These are never l-values; just use the aggregate emission code.
  case CK_NonAtomicToAtomic:
  case CK_AtomicToNonAtomic:
    assert(0 && "NYI");

  case CK_Dynamic: {
    LValue lv = buildLValue(e->getSubExpr());
    Address v = lv.getAddress();
    const auto *dce = cast<CXXDynamicCastExpr>(e);
    return MakeNaturalAlignAddrLValue(buildDynamicCast(v, dce), e->getType());
  }

  case CK_ConstructorConversion:
  case CK_UserDefinedConversion:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_LValueToRValue:
    return buildLValue(e->getSubExpr());

  case CK_NoOp: {
    // CK_NoOp can model a qualification conversion, which can remove an array
    // bound and change the IR type.
    LValue lv = buildLValue(e->getSubExpr());
    if (lv.isSimple()) {
      Address v = lv.getAddress();
      if (v.isValid()) {
        auto t = getTypes().convertTypeForMem(e->getType());
        if (v.getElementType() != t)
          lv.setAddress(
              builder.createElementBitCast(getLoc(e->getSourceRange()), v, t));
      }
    }
    return lv;
  }

  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase: {
    const auto *derivedClassTy =
        e->getSubExpr()->getType()->castAs<RecordType>();
    auto *derivedClassDecl = cast<CXXRecordDecl>(derivedClassTy->getDecl());

    LValue lv = buildLValue(e->getSubExpr());
    Address This = lv.getAddress();

    // Perform the derived-to-base conversion
    Address base = getAddressOfBaseClass(
        This, derivedClassDecl, e->path_begin(), e->path_end(),
        /*NullCheckValue=*/false, e->getExprLoc());

    // TODO: Support accesses to members of base classes in TBAA. For now, we
    // conservatively pretend that the complete object is of the base class
    // type.
    assert(!MissingFeatures::tbaa());
    return makeAddrLValue(base, e->getType(), lv.getBaseInfo());
  }
  case CK_ToUnion:
    assert(0 && "NYI");
  case CK_BaseToDerived: {
    assert(0 && "NYI");
  }
  case CK_LValueBitCast: {
    assert(0 && "NYI");
  }
  case CK_AddressSpaceConversion: {
    LValue lv = buildLValue(e->getSubExpr());
    QualType destTy = getContext().getPointerType(e->getType());
    auto srcAs =
        builder.getAddrSpaceAttr(e->getSubExpr()->getType().getAddressSpace());
    auto destAs = builder.getAddrSpaceAttr(e->getType().getAddressSpace());
    mlir::Value v = getTargetHooks().performAddrSpaceCast(
        *this, lv.getPointer(), srcAs, destAs, ConvertType(destTy));
    assert(!MissingFeatures::tbaa());
    return makeAddrLValue(Address(v, getTypes().convertTypeForMem(e->getType()),
                                  lv.getAddress().getAlignment()),
                          e->getType(), lv.getBaseInfo());
  }
  case CK_ObjCObjectLValueCast: {
    assert(0 && "NYI");
  }
  case CK_ZeroToOCLOpaqueType:
    llvm_unreachable("NULL to OpenCL opaque type lvalue cast is not valid");
  }

  llvm_unreachable("Unhandled lvalue cast kind?");
}

// TODO(cir): candidate for common helper between LLVM and CIR codegen.
static DeclRefExpr *tryToConvertMemberExprToDeclRefExpr(CIRGenFunction &cgf,
                                                        const MemberExpr *me) {
  if (auto *vd = dyn_cast<VarDecl>(me->getMemberDecl())) {
    // Try to emit static variable member expressions as DREs.
    return DeclRefExpr::Create(
        cgf.getContext(), NestedNameSpecifierLoc(), SourceLocation(), vd,
        /*RefersToEnclosingVariableOrCapture=*/false, me->getExprLoc(),
        me->getType(), me->getValueKind(), nullptr, nullptr, me->isNonOdrUse());
  }
  return nullptr;
}

LValue CIRGenFunction::buildCheckedLValue(const Expr *e, TypeCheckKind tck) {
  LValue lv;
  if (SanOpts.has(SanitizerKind::ArrayBounds) && isa<ArraySubscriptExpr>(e))
    assert(0 && "not implemented");
  else
    lv = buildLValue(e);
  if (!isa<DeclRefExpr>(e) && !lv.isBitField() && lv.isSimple()) {
    SanitizerSet skippedChecks;
    if (const auto *me = dyn_cast<MemberExpr>(e)) {
      bool isBaseCxxThis = isWrappedCXXThis(me->getBase());
      if (isBaseCxxThis)
        skippedChecks.set(SanitizerKind::Alignment, true);
      if (isBaseCxxThis || isa<DeclRefExpr>(me->getBase()))
        skippedChecks.set(SanitizerKind::Null, true);
    }
    buildTypeCheck(tck, e->getExprLoc(), lv.getPointer(), e->getType(),
                   lv.getAlignment(), skippedChecks);
  }
  return lv;
}

// TODO(cir): candidate for common AST helper for LLVM and CIR codegen
bool CIRGenFunction::isWrappedCXXThis(const Expr *obj) {
  const Expr *base = obj;
  while (!isa<CXXThisExpr>(base)) {
    // The result of a dynamic_cast can be null.
    if (isa<CXXDynamicCastExpr>(base))
      return false;

    if (const auto *ce = dyn_cast<CastExpr>(base)) {
      base = ce->getSubExpr();
    } else if (const auto *pe = dyn_cast<ParenExpr>(base)) {
      base = pe->getSubExpr();
    } else if (const auto *uo = dyn_cast<UnaryOperator>(base)) {
      if (uo->getOpcode() == UO_Extension)
        base = uo->getSubExpr();
      else
        return false;
    } else {
      return false;
    }
  }
  return true;
}

LValue CIRGenFunction::buildMemberExpr(const MemberExpr *e) {
  if (DeclRefExpr *dre = tryToConvertMemberExprToDeclRefExpr(*this, e)) {
    buildIgnoredExpr(e->getBase());
    return buildDeclRefLValue(dre);
  }

  Expr *baseExpr = e->getBase();
  // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a scalar.
  LValue baseLv;
  if (e->isArrow()) {
    LValueBaseInfo baseInfo;
    Address addr = buildPointerWithAlignment(baseExpr, &baseInfo);
    QualType ptrTy = baseExpr->getType()->getPointeeType();
    SanitizerSet skippedChecks;
    bool isBaseCxxThis = isWrappedCXXThis(baseExpr);
    if (isBaseCxxThis)
      skippedChecks.set(SanitizerKind::Alignment, true);
    if (isBaseCxxThis || isa<DeclRefExpr>(baseExpr))
      skippedChecks.set(SanitizerKind::Null, true);
    buildTypeCheck(TCK_MemberAccess, e->getExprLoc(), addr.getPointer(), ptrTy,
                   /*Alignment=*/CharUnits::Zero(), skippedChecks);
    baseLv = makeAddrLValue(addr, ptrTy, baseInfo);
  } else
    baseLv = buildCheckedLValue(baseExpr, TCK_MemberAccess);

  NamedDecl *nd = e->getMemberDecl();
  if (auto *field = dyn_cast<FieldDecl>(nd)) {
    LValue lv = buildLValueForField(baseLv, field);
    assert(!MissingFeatures::setObjCGCLValueClass() && "NYI");
    if (getLangOpts().OpenMP) {
      // If the member was explicitly marked as nontemporal, mark it as
      // nontemporal. If the base lvalue is marked as nontemporal, mark access
      // to children as nontemporal too.
      assert(0 && "not implemented");
    }
    return lv;
  }

  if (const auto *fd = dyn_cast<FunctionDecl>(nd))
    assert(0 && "not implemented");

  llvm_unreachable("Unhandled member declaration!");
}

LValue CIRGenFunction::buildCallExprLValue(const CallExpr *e) {
  RValue rv = buildCallExpr(e);

  if (!rv.isScalar())
    return makeAddrLValue(rv.getAggregateAddress(), e->getType(),
                          AlignmentSource::Decl);

  assert(e->getCallReturnType(getContext())->isReferenceType() &&
         "Can't have a scalar return unless the return type is a "
         "reference type!");

  return MakeNaturalAlignPointeeAddrLValue(rv.getScalarVal(), e->getType());
}

/// Evaluate an expression into a given memory location.
void CIRGenFunction::buildAnyExprToMem(const Expr *e, Address location,
                                       Qualifiers quals, bool isInit) {
  // FIXME: This function should take an LValue as an argument.
  switch (getEvaluationKind(e->getType())) {
  case TEK_Complex:
    assert(0 && "NYI");
    return;

  case TEK_Aggregate: {
    buildAggExpr(e, AggValueSlot::forAddr(location, quals,
                                          AggValueSlot::IsDestructed_t(isInit),
                                          AggValueSlot::DoesNotNeedGCBarriers,
                                          AggValueSlot::IsAliased_t(!isInit),
                                          AggValueSlot::MayOverlap));
    return;
  }

  case TEK_Scalar: {
    RValue rv = RValue::get(buildScalarExpr(e));
    LValue lv = makeAddrLValue(location, e->getType());
    buildStoreThroughLValue(rv, lv);
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}

static Address createReferenceTemporary(CIRGenFunction &cgf,
                                        const MaterializeTemporaryExpr *m,
                                        const Expr *inner,
                                        Address *alloca = nullptr) {
  // TODO(cir): CGF.getTargetHooks();
  switch (m->getStorageDuration()) {
  case SD_FullExpression:
  case SD_Automatic: {
    // TODO(cir): probably not needed / too LLVM specific?
    // If we have a constant temporary array or record try to promote it into a
    // constant global under the same rules a normal constant would've been
    // promoted. This is easier on the optimizer and generally emits fewer
    // instructions.
    QualType ty = inner->getType();
    if (cgf.CGM.getCodeGenOpts().MergeAllConstants &&
        (ty->isArrayType() || ty->isRecordType()) &&
        cgf.CGM.isTypeConstant(ty, /*ExcludeCtor=*/true, /*ExcludeDtor=*/false))
      assert(0 && "NYI");

    // The temporary memory should be created in the same scope as the extending
    // declaration of the temporary materialization expression.
    mlir::cir::AllocaOp extDeclAlloca;
    if (const clang::ValueDecl *extDecl = m->getExtendingDecl()) {
      auto extDeclAddrIter = cgf.LocalDeclMap.find(extDecl);
      if (extDeclAddrIter != cgf.LocalDeclMap.end()) {
        extDeclAlloca = dyn_cast_if_present<mlir::cir::AllocaOp>(
            extDeclAddrIter->second.getDefiningOp());
      }
    }
    mlir::OpBuilder::InsertPoint ip;
    if (extDeclAlloca)
      ip = {extDeclAlloca->getBlock(), extDeclAlloca->getIterator()};
    return cgf.CreateMemTemp(ty, cgf.getLoc(m->getSourceRange()),
                             cgf.getCounterRefTmpAsString(), alloca, ip);
  }
  case SD_Thread:
  case SD_Static:
    assert(0 && "NYI");

  case SD_Dynamic:
    llvm_unreachable("temporary can't have dynamic storage duration");
  }
  llvm_unreachable("unknown storage duration");
}

static void pushTemporaryCleanup(CIRGenFunction &cgf,
                                 const MaterializeTemporaryExpr *m,
                                 const Expr *e, Address referenceTemporary) {
  // Objective-C++ ARC:
  //   If we are binding a reference to a temporary that has ownership, we
  //   need to perform retain/release operations on the temporary.
  //
  // FIXME: This should be looking at E, not M.
  if (auto lifetime = m->getType().getObjCLifetime()) {
    assert(0 && "NYI");
  }

  CXXDestructorDecl *referenceTemporaryDtor = nullptr;
  if (const RecordType *rt =
          e->getType()->getBaseElementTypeUnsafe()->getAs<RecordType>()) {
    // Get the destructor for the reference temporary.
    auto *classDecl = cast<CXXRecordDecl>(rt->getDecl());
    if (!classDecl->hasTrivialDestructor())
      referenceTemporaryDtor = classDecl->getDestructor();
  }

  if (!referenceTemporaryDtor)
    return;

  // Call the destructor for the temporary.
  switch (m->getStorageDuration()) {
  case SD_Static:
  case SD_Thread: {
    if (e->getType()->isArrayType()) {
      llvm_unreachable("SD_Static|SD_Thread + array types not implemented");
    } else {
      llvm_unreachable("SD_Static|SD_Thread for general types not implemented");
    }
    llvm_unreachable("SD_Static|SD_Thread not implemented");
  }

  case SD_FullExpression:
    cgf.pushDestroy(NormalAndEHCleanup, referenceTemporary, e->getType(),
                    CIRGenFunction::destroyCXXObject,
                    cgf.getLangOpts().Exceptions);
    break;

  case SD_Automatic:
    llvm_unreachable("SD_Automatic not implemented");
    break;

  case SD_Dynamic:
    llvm_unreachable("temporary cannot have dynamic storage duration");
  }
}

LValue CIRGenFunction::buildMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *m) {
  const Expr *e = m->getSubExpr();

  assert((!m->getExtendingDecl() || !isa<VarDecl>(m->getExtendingDecl()) ||
          !cast<VarDecl>(m->getExtendingDecl())->isARCPseudoStrong()) &&
         "Reference should never be pseudo-strong!");

  // FIXME: ideally this would use buildAnyExprToMem, however, we cannot do so
  // as that will cause the lifetime adjustment to be lost for ARC
  auto ownership = m->getType().getObjCLifetime();
  if (ownership != Qualifiers::OCL_None &&
      ownership != Qualifiers::OCL_ExplicitNone) {
    assert(0 && "NYI");
  }

  SmallVector<const Expr *, 2> commaLhSs;
  SmallVector<SubobjectAdjustment, 2> adjustments;
  e = e->skipRValueSubobjectAdjustments(commaLhSs, adjustments);

  for (const auto &ignored : commaLhSs)
    buildIgnoredExpr(ignored);

  if (const auto *opaque = dyn_cast<OpaqueValueExpr>(e))
    assert(0 && "NYI");

  // Create and initialize the reference temporary.
  Address alloca = Address::invalid();
  Address object = createReferenceTemporary(*this, m, e, &alloca);

  if (auto var =
          dyn_cast<mlir::cir::GlobalOp>(object.getPointer().getDefiningOp())) {
    // TODO(cir): add something akin to stripPointerCasts() to ptr above
    assert(0 && "NYI");
  } else {
    switch (m->getStorageDuration()) {
    case SD_Automatic:
      assert(!MissingFeatures::shouldEmitLifetimeMarkers());
      break;

    case SD_FullExpression: {
      if (!ShouldEmitLifetimeMarkers)
        break;
      assert(0 && "NYI");
      break;
    }

    default:
      break;
    }

    buildAnyExprToMem(e, object, Qualifiers(), /*IsInit*/ true);
  }
  pushTemporaryCleanup(*this, m, e, object);

  // Perform derived-to-base casts and/or field accesses, to get from the
  // temporary object we created (and, potentially, for which we extended
  // the lifetime) to the subobject we're binding the reference to.
  for (SubobjectAdjustment &adjustment : llvm::reverse(adjustments)) {
    (void)adjustment;
    assert(0 && "NYI");
  }

  return makeAddrLValue(object, m->getType(), AlignmentSource::Decl);
}

LValue CIRGenFunction::buildOpaqueValueLValue(const OpaqueValueExpr *e) {
  assert(OpaqueValueMappingData::shouldBindAsLValue(e));
  return getOrCreateOpaqueLValueMapping(e);
}

LValue
CIRGenFunction::getOrCreateOpaqueLValueMapping(const OpaqueValueExpr *e) {
  assert(OpaqueValueMapping::shouldBindAsLValue(e));

  llvm::DenseMap<const OpaqueValueExpr *, LValue>::iterator it =
      OpaqueLValues.find(e);

  if (it != OpaqueLValues.end())
    return it->second;

  assert(e->isUnique() && "LValue for a nonunique OVE hasn't been emitted");
  return buildLValue(e->getSourceExpr());
}

RValue
CIRGenFunction::getOrCreateOpaqueRValueMapping(const OpaqueValueExpr *e) {
  assert(!OpaqueValueMapping::shouldBindAsLValue(e));

  llvm::DenseMap<const OpaqueValueExpr *, RValue>::iterator it =
      OpaqueRValues.find(e);

  if (it != OpaqueRValues.end())
    return it->second;

  assert(e->isUnique() && "RValue for a nonunique OVE hasn't been emitted");
  return buildAnyExpr(e->getSourceExpr());
}

namespace {
// Handle the case where the condition is a constant evaluatable simple integer,
// which means we don't have to separately handle the true/false blocks.
std::optional<LValue> handleConditionalOperatorLValueSimpleCase(
    CIRGenFunction &cgf, const AbstractConditionalOperator *e) {
  const Expr *condExpr = e->getCond();
  bool condExprBool;
  if (cgf.ConstantFoldsToSimpleInteger(condExpr, condExprBool)) {
    const Expr *live = e->getTrueExpr(), *dead = e->getFalseExpr();
    if (!condExprBool)
      std::swap(live, dead);

    if (!cgf.ContainsLabel(dead)) {
      // If the true case is live, we need to track its region.
      if (condExprBool) {
        assert(!MissingFeatures::incrementProfileCounter());
      }
      // If a throw expression we emit it and return an undefined lvalue
      // because it can't be used.
      if (auto *throwExpr = dyn_cast<CXXThrowExpr>(live->IgnoreParens())) {
        llvm_unreachable("NYI");
      }
      return cgf.buildLValue(live);
    }
  }
  return std::nullopt;
}
} // namespace

/// Emit the operand of a glvalue conditional operator. This is either a glvalue
/// or a (possibly-parenthesized) throw-expression. If this is a throw, no
/// LValue is returned and the current block has been terminated.
static std::optional<LValue> buildLValueOrThrowExpression(CIRGenFunction &cgf,
                                                          const Expr *operand) {
  if (auto *throwExpr = dyn_cast<CXXThrowExpr>(operand->IgnoreParens())) {
    llvm_unreachable("NYI");
  }

  return cgf.buildLValue(operand);
}

// Create and generate the 3 blocks for a conditional operator.
// Leaves the 'current block' in the continuation basic block.
template <typename FuncTy>
CIRGenFunction::ConditionalInfo
CIRGenFunction::buildConditionalBlocks(const AbstractConditionalOperator *e,
                                       const FuncTy &branchGenFunc) {
  ConditionalInfo info;
  auto &cgf = *this;
  ConditionalEvaluation eval(cgf);
  auto loc = cgf.getLoc(e->getSourceRange());
  auto &builder = cgf.getBuilder();
  auto *trueExpr = e->getTrueExpr();
  auto *falseExpr = e->getFalseExpr();

  mlir::Value condV = cgf.buildOpOnBoolExpr(loc, e->getCond());
  SmallVector<mlir::OpBuilder::InsertPoint, 2> insertPoints{};
  mlir::Type yieldTy{};

  auto patchVoidOrThrowSites = [&]() {
    if (insertPoints.empty())
      return;
    // If both arms are void, so be it.
    if (!yieldTy)
      yieldTy = cgf.VoidTy;

    // Insert required yields.
    for (auto &toInsert : insertPoints) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.restoreInsertionPoint(toInsert);

      // Block does not return: build empty yield.
      if (mlir::isa<mlir::cir::VoidType>(yieldTy)) {
        builder.create<mlir::cir::YieldOp>(loc);
      } else { // Block returns: set null yield value.
        mlir::Value op0 = builder.getNullValue(yieldTy, loc);
        builder.create<mlir::cir::YieldOp>(loc, op0);
      }
    }
  };

  info.Result = builder
                    .create<mlir::cir::TernaryOp>(
                        loc, condV, /*trueBuilder=*/
                        [&](mlir::OpBuilder &b, mlir::Location loc) {
                          CIRGenFunction::LexicalScope lexScope{
                              *this, loc, b.getInsertionBlock()};
                          cgf.currLexScope->setAsTernary();

                          assert(!MissingFeatures::incrementProfileCounter());
                          eval.begin(cgf);
                          info.LHS = branchGenFunc(cgf, trueExpr);
                          auto lhs = info.LHS->getPointer();
                          eval.end(cgf);

                          if (lhs) {
                            yieldTy = lhs.getType();
                            b.create<mlir::cir::YieldOp>(loc, lhs);
                            return;
                          }
                          // If LHS or RHS is a throw or void expression we need
                          // to patch arms as to properly match yield types.
                          insertPoints.push_back(b.saveInsertionPoint());
                        },
                        /*falseBuilder=*/
                        [&](mlir::OpBuilder &b, mlir::Location loc) {
                          CIRGenFunction::LexicalScope lexScope{
                              *this, loc, b.getInsertionBlock()};
                          cgf.currLexScope->setAsTernary();

                          assert(!MissingFeatures::incrementProfileCounter());
                          eval.begin(cgf);
                          info.RHS = branchGenFunc(cgf, falseExpr);
                          auto rhs = info.RHS->getPointer();
                          eval.end(cgf);

                          if (rhs) {
                            yieldTy = rhs.getType();
                            b.create<mlir::cir::YieldOp>(loc, rhs);
                          } else {
                            // If LHS or RHS is a throw or void expression we
                            // need to patch arms as to properly match yield
                            // types.
                            insertPoints.push_back(b.saveInsertionPoint());
                          }

                          patchVoidOrThrowSites();
                        })
                    .getResult();
  return info;
}

LValue CIRGenFunction::buildConditionalOperatorLValue(
    const AbstractConditionalOperator *expr) {
  if (!expr->isGLValue()) {
    // ?: here should be an aggregate.
    assert(hasAggregateEvaluationKind(expr->getType()) &&
           "Unexpected conditional operator!");
    return buildAggExprToLValue(expr);
  }

  OpaqueValueMapping binding(*this, expr);
  if (std::optional<LValue> res =
          handleConditionalOperatorLValueSimpleCase(*this, expr))
    return *res;

  ConditionalInfo info =
      buildConditionalBlocks(expr, [](CIRGenFunction &cgf, const Expr *e) {
        return buildLValueOrThrowExpression(cgf, e);
      });

  if ((info.LHS && !info.LHS->isSimple()) ||
      (info.RHS && !info.RHS->isSimple()))
    llvm_unreachable("unsupported conditional operator");

  if (info.LHS && info.RHS) {
    Address lhsAddr = info.LHS->getAddress();
    Address rhsAddr = info.RHS->getAddress();
    Address result(info.Result, lhsAddr.getElementType(),
                   std::min(lhsAddr.getAlignment(), rhsAddr.getAlignment()));
    AlignmentSource alignSource =
        std::max(info.LHS->getBaseInfo().getAlignmentSource(),
                 info.RHS->getBaseInfo().getAlignmentSource());
    assert(!MissingFeatures::tbaa());
    return makeAddrLValue(result, expr->getType(), LValueBaseInfo(alignSource));
  }
  llvm_unreachable("NYI");
}

/// Emit code to compute a designator that specifies the location
/// of the expression.
/// FIXME: document this function better.
LValue CIRGenFunction::buildLValue(const Expr *e) {
  // FIXME: ApplyDebugLocation DL(*this, E);
  switch (e->getStmtClass()) {
  default: {
    emitError(getLoc(e->getExprLoc()), "l-value not implemented for '")
        << e->getStmtClassName() << "'";
    assert(0 && "not implemented");
  }
  case Expr::ConditionalOperatorClass:
    return buildConditionalOperatorLValue(cast<ConditionalOperator>(e));
  case Expr::ArraySubscriptExprClass:
    return buildArraySubscriptExpr(cast<ArraySubscriptExpr>(e));
  case Expr::ExtVectorElementExprClass:
    return buildExtVectorElementExpr(cast<ExtVectorElementExpr>(e));
  case Expr::BinaryOperatorClass:
    return buildBinaryOperatorLValue(cast<BinaryOperator>(e));
  case Expr::CompoundAssignOperatorClass: {
    QualType ty = e->getType();
    if (const AtomicType *at = ty->getAs<AtomicType>())
      assert(0 && "not yet implemented");
    if (!ty->isAnyComplexType())
      return buildCompoundAssignmentLValue(cast<CompoundAssignOperator>(e));
    return buildComplexCompoundAssignmentLValue(
        cast<CompoundAssignOperator>(e));
  }
  case Expr::CallExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXOperatorCallExprClass:
  case Expr::UserDefinedLiteralClass:
    return buildCallExprLValue(cast<CallExpr>(e));
  case Expr::ExprWithCleanupsClass: {
    const auto *cleanups = cast<ExprWithCleanups>(e);
    LValue lv;

    auto scopeLoc = getLoc(e->getSourceRange());
    [[maybe_unused]] auto scope = builder.create<mlir::cir::ScopeOp>(
        scopeLoc, /*scopeBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          CIRGenFunction::LexicalScope lexScope{*this, loc,
                                                builder.getInsertionBlock()};

          lv = buildLValue(cleanups->getSubExpr());
          if (lv.isSimple()) {
            // Defend against branches out of gnu statement expressions
            // surrounded by cleanups.
            Address addr = lv.getAddress();
            auto v = addr.getPointer();
            lv = LValue::makeAddr(addr.withPointer(v, NotKnownNonNull),
                                  lv.getType(), getContext(), lv.getBaseInfo(),
                                  lv.getTBAAInfo());
          }
        });

    // FIXME: Is it possible to create an ExprWithCleanups that produces a
    // bitfield lvalue or some other non-simple lvalue?
    return lv;
  }
  case Expr::ParenExprClass:
    return buildLValue(cast<ParenExpr>(e)->getSubExpr());
  case Expr::DeclRefExprClass:
    return buildDeclRefLValue(cast<DeclRefExpr>(e));
  case Expr::UnaryOperatorClass:
    return buildUnaryOpLValue(cast<UnaryOperator>(e));
  case Expr::StringLiteralClass:
    return buildStringLiteralLValue(cast<StringLiteral>(e));
  case Expr::MemberExprClass:
    return buildMemberExpr(cast<MemberExpr>(e));
  case Expr::CompoundLiteralExprClass:
    return buildCompoundLiteralLValue(cast<CompoundLiteralExpr>(e));
  case Expr::PredefinedExprClass:
    return buildPredefinedLValue(cast<PredefinedExpr>(e));
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass:
  case Expr::CXXAddrspaceCastExprClass:
  case Expr::ObjCBridgedCastExprClass:
    emitError(getLoc(e->getExprLoc()), "l-value not implemented for '")
        << e->getStmtClassName() << "'";
    assert(0 && "Use buildCastLValue below, remove me when adding testcase");
  case Expr::CStyleCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::ImplicitCastExprClass:
    return buildCastLValue(cast<CastExpr>(e));
  case Expr::OpaqueValueExprClass:
    return buildOpaqueValueLValue(cast<OpaqueValueExpr>(e));

  case Expr::MaterializeTemporaryExprClass:
    return buildMaterializeTemporaryExpr(cast<MaterializeTemporaryExpr>(e));

  case Expr::ObjCPropertyRefExprClass:
    llvm_unreachable("cannot emit a property reference directly");
  case Expr::StmtExprClass:
    return buildStmtExprLValue(cast<StmtExpr>(e));
  }

  return LValue::makeAddr(Address::invalid(), e->getType());
}

/// Given the address of a temporary variable, produce an r-value of its type.
RValue CIRGenFunction::convertTempToRValue(Address addr, clang::QualType type,
                                           clang::SourceLocation loc) {
  LValue lvalue = makeAddrLValue(addr, type, AlignmentSource::Decl);
  switch (getEvaluationKind(type)) {
  case TEK_Complex:
    llvm_unreachable("NYI");
  case TEK_Aggregate:
    llvm_unreachable("NYI");
  case TEK_Scalar:
    return RValue::get(buildLoadOfScalar(lvalue, loc));
  }
  llvm_unreachable("NYI");
}

/// An LValue is a candidate for having its loads and stores be made atomic if
/// we are operating under /volatile:ms *and* the LValue itself is volatile and
/// performing such an operation can be performed without a libcall.
bool CIRGenFunction::LValueIsSuitableForInlineAtomic(LValue lv) {
  if (!CGM.getLangOpts().MSVolatile)
    return false;

  llvm_unreachable("NYI");
}

/// Emit an `if` on a boolean condition, filling `then` and `else` into
/// appropriated regions.
mlir::LogicalResult CIRGenFunction::buildIfOnBoolExpr(const Expr *cond,
                                                      const Stmt *thenS,
                                                      const Stmt *elseS) {
  // Attempt to be more accurate as possible with IfOp location, generate
  // one fused location that has either 2 or 4 total locations, depending
  // on else's availability.
  auto getStmtLoc = [this](const Stmt &s) {
    return mlir::FusedLoc::get(builder.getContext(),
                               {getLoc(s.getSourceRange().getBegin()),
                                getLoc(s.getSourceRange().getEnd())});
  };
  auto thenLoc = getStmtLoc(*thenS);
  std::optional<mlir::Location> elseLoc;
  if (elseS)
    elseLoc = getStmtLoc(*elseS);

  mlir::LogicalResult resThen = mlir::success(), resElse = mlir::success();
  buildIfOnBoolExpr(
      cond, /*thenBuilder=*/
      [&](mlir::OpBuilder &, mlir::Location) {
        LexicalScope lexScope{*this, thenLoc, builder.getInsertionBlock()};
        resThen = buildStmt(thenS, /*useCurrentScope=*/true);
      },
      thenLoc,
      /*elseBuilder=*/
      [&](mlir::OpBuilder &, mlir::Location) {
        assert(elseLoc && "Invalid location for elseS.");
        LexicalScope lexScope{*this, *elseLoc, builder.getInsertionBlock()};
        resElse = buildStmt(elseS, /*useCurrentScope=*/true);
      },
      elseLoc);

  return mlir::LogicalResult::success(resThen.succeeded() &&
                                      resElse.succeeded());
}

/// Emit an `if` on a boolean condition, filling `then` and `else` into
/// appropriated regions.
mlir::cir::IfOp CIRGenFunction::buildIfOnBoolExpr(
    const clang::Expr *cond,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> thenBuilder,
    mlir::Location thenLoc,
    llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)> elseBuilder,
    std::optional<mlir::Location> elseLoc) {

  SmallVector<mlir::Location, 2> ifLocs{thenLoc};
  if (elseLoc)
    ifLocs.push_back(*elseLoc);
  auto loc = mlir::FusedLoc::get(builder.getContext(), ifLocs);

  // Emit the code with the fully general case.
  mlir::Value condV = buildOpOnBoolExpr(loc, cond);
  return builder.create<mlir::cir::IfOp>(loc, condV, elseLoc.has_value(),
                                         /*thenBuilder=*/thenBuilder,
                                         /*elseBuilder=*/elseBuilder);
}

/// TODO(cir): PGO data
/// TODO(cir): see EmitBranchOnBoolExpr for extra ideas).
mlir::Value CIRGenFunction::buildOpOnBoolExpr(mlir::Location loc,
                                              const Expr *cond) {
  // TODO(CIR): scoped ApplyDebugLocation DL(*this, Cond);
  // TODO(CIR): __builtin_unpredictable and profile counts?
  cond = cond->IgnoreParens();

  // if (const BinaryOperator *CondBOp = dyn_cast<BinaryOperator>(cond)) {
  //   llvm_unreachable("binaryoperator ifstmt NYI");
  // }

  if (const UnaryOperator *condUOp = dyn_cast<UnaryOperator>(cond)) {
    // In LLVM the condition is reversed here for efficient codegen.
    // This should be done in CIR prior to LLVM lowering, if we do now
    // we can make CIR based diagnostics misleading.
    //  cir.ternary(!x, t, f) -> cir.ternary(x, f, t)
    assert(!MissingFeatures::shouldReverseUnaryCondOnBoolExpr());
  }

  if (const ConditionalOperator *condOp = dyn_cast<ConditionalOperator>(cond)) {
    auto *trueExpr = condOp->getTrueExpr();
    auto *falseExpr = condOp->getFalseExpr();
    mlir::Value condV = buildOpOnBoolExpr(loc, condOp->getCond());

    auto ternaryOpRes =
        builder
            .create<mlir::cir::TernaryOp>(
                loc, condV, /*thenBuilder=*/
                [this, trueExpr](mlir::OpBuilder &b, mlir::Location loc) {
                  auto lhs = buildScalarExpr(trueExpr);
                  b.create<mlir::cir::YieldOp>(loc, lhs);
                },
                /*elseBuilder=*/
                [this, falseExpr](mlir::OpBuilder &b, mlir::Location loc) {
                  auto rhs = buildScalarExpr(falseExpr);
                  b.create<mlir::cir::YieldOp>(loc, rhs);
                })
            .getResult();

    return buildScalarConversion(ternaryOpRes, condOp->getType(),
                                 getContext().BoolTy, condOp->getExprLoc());
  }

  if (const CXXThrowExpr *Throw = dyn_cast<CXXThrowExpr>(cond)) {
    llvm_unreachable("NYI");
  }

  // If the branch has a condition wrapped by __builtin_unpredictable,
  // create metadata that specifies that the branch is unpredictable.
  // Don't bother if not optimizing because that metadata would not be used.
  auto *call = dyn_cast<CallExpr>(cond->IgnoreImpCasts());
  if (call && CGM.getCodeGenOpts().OptimizationLevel != 0) {
    assert(!MissingFeatures::insertBuiltinUnpredictable());
  }

  // Emit the code with the fully general case.
  return evaluateExprAsBool(cond);
}

mlir::Value CIRGenFunction::buildAlloca(StringRef name, mlir::Type ty,
                                        mlir::Location loc, CharUnits alignment,
                                        bool insertIntoFnEntryBlock,
                                        mlir::Value arraySize) {
  mlir::Block *entryBlock = insertIntoFnEntryBlock
                                ? getCurFunctionEntryBlock()
                                : currLexScope->getEntryBlock();

  // If this is an alloca in the entry basic block of a cir.try and there's
  // a surrounding cir.scope, make sure the alloca ends up in the surrounding
  // scope instead. This is necessary in order to guarantee all SSA values are
  // reachable during cleanups.
  if (auto tryOp = llvm::dyn_cast_if_present<mlir::cir::TryOp>(
          entryBlock->getParentOp())) {
    if (auto scopeOp = llvm::dyn_cast<mlir::cir::ScopeOp>(tryOp->getParentOp()))
      entryBlock = &scopeOp.getRegion().front();
  }

  return buildAlloca(name, ty, loc, alignment,
                     builder.getBestAllocaInsertPoint(entryBlock), arraySize);
}

mlir::Value CIRGenFunction::buildAlloca(StringRef name, mlir::Type ty,
                                        mlir::Location loc, CharUnits alignment,
                                        mlir::OpBuilder::InsertPoint ip,
                                        mlir::Value arraySize) {
  // CIR uses its own alloca AS rather than follow the target data layout like
  // original CodeGen. The data layout awareness should be done in the lowering
  // pass instead.
  auto localVarPtrTy = builder.getPointerTo(ty, getCIRAllocaAddressSpace());
  auto alignIntAttr = CGM.getSize(alignment);

  mlir::Value addr;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(ip);
    addr = builder.createAlloca(loc, /*addr type*/ localVarPtrTy,
                                /*var type*/ ty, name, alignIntAttr, arraySize);
    if (currVarDecl) {
      auto alloca = cast<mlir::cir::AllocaOp>(addr.getDefiningOp());
      alloca.setAstAttr(ASTVarDeclAttr::get(builder.getContext(), currVarDecl));
    }
  }
  return addr;
}

mlir::Value CIRGenFunction::buildAlloca(StringRef name, QualType ty,
                                        mlir::Location loc, CharUnits alignment,
                                        bool insertIntoFnEntryBlock,
                                        mlir::Value arraySize) {
  return buildAlloca(name, getCIRType(ty), loc, alignment,
                     insertIntoFnEntryBlock, arraySize);
}

mlir::Value CIRGenFunction::buildLoadOfScalar(LValue lvalue,
                                              SourceLocation loc) {
  return buildLoadOfScalar(lvalue.getAddress(), lvalue.isVolatile(),
                           lvalue.getType(), getLoc(loc), lvalue.getBaseInfo(),
                           lvalue.getTBAAInfo(), lvalue.isNontemporal());
}

mlir::Value CIRGenFunction::buildLoadOfScalar(LValue lvalue,
                                              mlir::Location loc) {
  return buildLoadOfScalar(lvalue.getAddress(), lvalue.isVolatile(),
                           lvalue.getType(), loc, lvalue.getBaseInfo(),
                           lvalue.getTBAAInfo(), lvalue.isNontemporal());
}

mlir::Value CIRGenFunction::buildFromMemory(mlir::Value value, QualType ty) {
  if (!ty->isBooleanType() && hasBooleanRepresentation(ty)) {
    llvm_unreachable("NIY");
  }

  return value;
}

mlir::Value CIRGenFunction::buildLoadOfScalar(Address addr, bool isVolatile,
                                              QualType ty, SourceLocation loc,
                                              LValueBaseInfo baseInfo,
                                              TBAAAccessInfo tbaaInfo,
                                              bool isNontemporal) {
  return buildLoadOfScalar(addr, isVolatile, ty, getLoc(loc), baseInfo,
                           tbaaInfo, isNontemporal);
}

mlir::Value CIRGenFunction::buildLoadOfScalar(Address addr, bool isVolatile,
                                              QualType ty, mlir::Location loc,
                                              LValueBaseInfo baseInfo,
                                              TBAAAccessInfo tbaaInfo,
                                              bool isNontemporal) {
  // Atomic operations have to be done on integral types
  LValue atomicLValue =
      LValue::makeAddr(addr, ty, getContext(), baseInfo, tbaaInfo);
  if (ty->isAtomicType() || LValueIsSuitableForInlineAtomic(atomicLValue)) {
    llvm_unreachable("NYI");
  }

  auto elemTy = addr.getElementType();

  if (const auto *clangVecTy = ty->getAs<clang::VectorType>()) {
    // Handle vectors of size 3 like size 4 for better performance.
    const auto vTy = cast<mlir::cir::VectorType>(elemTy);

    if (!CGM.getCodeGenOpts().PreserveVec3Type &&
        clangVecTy->getNumElements() == 3) {
      auto loc = addr.getPointer().getLoc();
      auto vec4Ty =
          mlir::cir::VectorType::get(vTy.getContext(), vTy.getEltType(), 4);
      Address cast = addr.withElementType(vec4Ty);
      // Now load value.
      mlir::Value v = builder.createLoad(loc, cast);

      // Shuffle vector to get vec3.
      v = builder.createVecShuffle(loc, v, ArrayRef<int64_t>{0, 1, 2});
      return buildFromMemory(v, ty);
    }
  }

  auto ptr = addr.getPointer();
  if (mlir::isa<mlir::cir::VoidType>(elemTy)) {
    elemTy = mlir::cir::IntType::get(builder.getContext(), 8, true);
    auto elemPtrTy = mlir::cir::PointerType::get(builder.getContext(), elemTy);
    ptr = builder.create<mlir::cir::CastOp>(loc, elemPtrTy,
                                            mlir::cir::CastKind::bitcast, ptr);
  }

  mlir::Value load = builder.CIRBaseBuilderTy::createLoad(loc, ptr, isVolatile);

  if (isNontemporal) {
    llvm_unreachable("NYI");
  }

  assert(!MissingFeatures::tbaa() && "NYI");
  assert(!MissingFeatures::emitScalarRangeCheck() && "NYI");

  return buildFromMemory(load, ty);
}

// Note: this function also emit constructor calls to support a MSVC extensions
// allowing explicit constructor function call.
RValue CIRGenFunction::buildCXXMemberCallExpr(const CXXMemberCallExpr *ce,
                                              ReturnValueSlot returnValue) {

  const Expr *callee = ce->getCallee()->IgnoreParens();

  if (isa<BinaryOperator>(callee))
    return buildCXXMemberPointerCallExpr(ce, returnValue);

  const auto *me = cast<MemberExpr>(callee);
  const auto *md = cast<CXXMethodDecl>(me->getMemberDecl());

  if (md->isStatic()) {
    llvm_unreachable("NYI");
  }

  bool hasQualifier = me->hasQualifier();
  NestedNameSpecifier *qualifier = hasQualifier ? me->getQualifier() : nullptr;
  bool isArrow = me->isArrow();
  const Expr *base = me->getBase();

  return buildCXXMemberOrOperatorMemberCallExpr(
      ce, md, returnValue, hasQualifier, qualifier, isArrow, base);
}

RValue CIRGenFunction::buildReferenceBindingToExpr(const Expr *e) {
  // Emit the expression as an lvalue.
  LValue lv = buildLValue(e);
  assert(lv.isSimple());
  auto value = lv.getPointer();

  if (sanitizePerformTypeCheck() && !e->getType()->isFunctionType()) {
    assert(0 && "NYI");
  }

  return RValue::get(value);
}

Address CIRGenFunction::buildLoadOfReference(LValue refLVal, mlir::Location loc,
                                             LValueBaseInfo *pointeeBaseInfo,
                                             TBAAAccessInfo *pointeeTBAAInfo) {
  assert(!refLVal.isVolatile() && "NYI");
  mlir::cir::LoadOp load = builder.create<mlir::cir::LoadOp>(
      loc, refLVal.getAddress().getElementType(),
      refLVal.getAddress().getPointer());

  // TODO(cir): DecorateInstructionWithTBAA relevant for us?
  assert(!MissingFeatures::tbaa());

  QualType pointeeType = refLVal.getType()->getPointeeType();
  CharUnits align =
      CGM.getNaturalTypeAlignment(pointeeType, pointeeBaseInfo, pointeeTBAAInfo,
                                  /* forPointeeType= */ true);
  return Address(load, getTypes().convertTypeForMem(pointeeType), align);
}

LValue CIRGenFunction::buildLoadOfReferenceLValue(LValue refLVal,
                                                  mlir::Location loc) {
  LValueBaseInfo pointeeBaseInfo;
  Address pointeeAddr = buildLoadOfReference(refLVal, loc, &pointeeBaseInfo);
  return makeAddrLValue(pointeeAddr, refLVal.getType()->getPointeeType(),
                        pointeeBaseInfo);
}

void CIRGenFunction::buildUnreachable(SourceLocation loc) {
  if (SanOpts.has(SanitizerKind::Unreachable))
    llvm_unreachable("NYI");
  builder.create<mlir::cir::UnreachableOp>(getLoc(loc));
}

//===----------------------------------------------------------------------===//
// CIR builder helpers
//===----------------------------------------------------------------------===//

Address CIRGenFunction::CreateMemTemp(QualType ty, mlir::Location loc,
                                      const Twine &name, Address *alloca,
                                      mlir::OpBuilder::InsertPoint ip) {
  // FIXME: Should we prefer the preferred type alignment here?
  return CreateMemTemp(ty, getContext().getTypeAlignInChars(ty), loc, name,
                       alloca, ip);
}

Address CIRGenFunction::CreateMemTemp(QualType ty, CharUnits align,
                                      mlir::Location loc, const Twine &name,
                                      Address *alloca,
                                      mlir::OpBuilder::InsertPoint ip) {
  Address result =
      CreateTempAlloca(getTypes().convertTypeForMem(ty), align, loc, name,
                       /*ArraySize=*/nullptr, alloca, ip);
  if (ty->isConstantMatrixType()) {
    assert(0 && "NYI");
  }
  return result;
}

/// This creates a alloca and inserts it into the entry block of the
/// current region.
Address CIRGenFunction::CreateTempAllocaWithoutCast(
    mlir::Type ty, CharUnits align, mlir::Location loc, const Twine &name,
    mlir::Value arraySize, mlir::OpBuilder::InsertPoint ip) {
  auto alloca = ip.isSet() ? CreateTempAlloca(ty, loc, name, ip, arraySize)
                           : CreateTempAlloca(ty, loc, name, arraySize);
  alloca.setAlignmentAttr(CGM.getSize(align));
  return Address(alloca, ty, align);
}

/// This creates a alloca and inserts it into the entry block. The alloca is
/// casted to default address space if necessary.
Address CIRGenFunction::CreateTempAlloca(mlir::Type ty, CharUnits align,
                                         mlir::Location loc, const Twine &name,
                                         mlir::Value arraySize,
                                         Address *allocaAddr,
                                         mlir::OpBuilder::InsertPoint ip) {
  auto alloca =
      CreateTempAllocaWithoutCast(ty, align, loc, name, arraySize, ip);
  if (allocaAddr)
    *allocaAddr = alloca;
  mlir::Value v = alloca.getPointer();
  // Alloca always returns a pointer in alloca address space, which may
  // be different from the type defined by the language. For example,
  // in C++ the auto variables are in the default address space. Therefore
  // cast alloca to the default address space when necessary.
  if (auto astas =
          builder.getAddrSpaceAttr(CGM.getLangTempAllocaAddressSpace());
      getCIRAllocaAddressSpace() != astas) {
    llvm_unreachable("Requires address space cast which is NYI");
  }
  return Address(v, ty, align);
}

/// This creates an alloca and inserts it into the entry block if \p ArraySize
/// is nullptr, otherwise inserts it at the current insertion point of the
/// builder.
mlir::cir::AllocaOp
CIRGenFunction::CreateTempAlloca(mlir::Type ty, mlir::Location loc,
                                 const Twine &name, mlir::Value arraySize,
                                 bool insertIntoFnEntryBlock) {
  return cast<mlir::cir::AllocaOp>(buildAlloca(name.str(), ty, loc, CharUnits(),
                                               insertIntoFnEntryBlock,
                                               arraySize)
                                       .getDefiningOp());
}

/// This creates an alloca and inserts it into the provided insertion point
mlir::cir::AllocaOp CIRGenFunction::CreateTempAlloca(
    mlir::Type ty, mlir::Location loc, const Twine &name,
    mlir::OpBuilder::InsertPoint ip, mlir::Value arraySize) {
  assert(ip.isSet() && "Insertion point is not set");
  return cast<mlir::cir::AllocaOp>(
      buildAlloca(name.str(), ty, loc, CharUnits(), ip, arraySize)
          .getDefiningOp());
}

/// Just like CreateTempAlloca above, but place the alloca into the function
/// entry basic block instead.
mlir::cir::AllocaOp CIRGenFunction::CreateTempAllocaInFnEntryBlock(
    mlir::Type ty, mlir::Location loc, const Twine &name,
    mlir::Value arraySize) {
  return CreateTempAlloca(ty, loc, name, arraySize,
                          /*insertIntoFnEntryBlock=*/true);
}

/// Given an object of the given canonical type, can we safely copy a
/// value out of it based on its initializer?
static bool isConstantEmittableObjectType(QualType type) {
  assert(type.isCanonical());
  assert(!type->isReferenceType());

  // Must be const-qualified but non-volatile.
  Qualifiers qs = type.getLocalQualifiers();
  if (!qs.hasConst() || qs.hasVolatile())
    return false;

  // Otherwise, all object types satisfy this except C++ classes with
  // mutable subobjects or non-trivial copy/destroy behavior.
  if (const auto *rt = dyn_cast<RecordType>(type))
    if (const auto *rd = dyn_cast<CXXRecordDecl>(rt->getDecl()))
      if (rd->hasMutableFields() || !rd->isTrivial())
        return false;

  return true;
}

/// Can we constant-emit a load of a reference to a variable of the
/// given type?  This is different from predicates like
/// Decl::mightBeUsableInConstantExpressions because we do want it to apply
/// in situations that don't necessarily satisfy the language's rules
/// for this (e.g. C++'s ODR-use rules).  For example, we want to able
/// to do this with const float variables even if those variables
/// aren't marked 'constexpr'.
enum ConstantEmissionKind {
  CEK_None,
  CEK_AsReferenceOnly,
  CEK_AsValueOrReference,
  CEK_AsValueOnly
};
static ConstantEmissionKind checkVarTypeForConstantEmission(QualType type) {
  type = type.getCanonicalType();
  if (const auto *ref = dyn_cast<ReferenceType>(type)) {
    if (isConstantEmittableObjectType(ref->getPointeeType()))
      return CEK_AsValueOrReference;
    return CEK_AsReferenceOnly;
  }
  if (isConstantEmittableObjectType(type))
    return CEK_AsValueOnly;
  return CEK_None;
}

/// Try to emit a reference to the given value without producing it as
/// an l-value.  This is just an optimization, but it avoids us needing
/// to emit global copies of variables if they're named without triggering
/// a formal use in a context where we can't emit a direct reference to them,
/// for instance if a block or lambda or a member of a local class uses a
/// const int variable or constexpr variable from an enclosing function.
CIRGenFunction::ConstantEmission
CIRGenFunction::tryEmitAsConstant(DeclRefExpr *refExpr) {
  ValueDecl *value = refExpr->getDecl();

  // The value needs to be an enum constant or a constant variable.
  ConstantEmissionKind cek;
  if (isa<ParmVarDecl>(value)) {
    cek = CEK_None;
  } else if (auto *var = dyn_cast<VarDecl>(value)) {
    cek = checkVarTypeForConstantEmission(var->getType());
  } else if (isa<EnumConstantDecl>(value)) {
    cek = CEK_AsValueOnly;
  } else {
    cek = CEK_None;
  }
  if (cek == CEK_None)
    return ConstantEmission();

  Expr::EvalResult result;
  bool resultIsReference;
  QualType resultType;

  // It's best to evaluate all the way as an r-value if that's permitted.
  if (cek != CEK_AsReferenceOnly &&
      refExpr->EvaluateAsRValue(result, getContext())) {
    resultIsReference = false;
    resultType = refExpr->getType();

    // Otherwise, try to evaluate as an l-value.
  } else if (cek != CEK_AsValueOnly &&
             refExpr->EvaluateAsLValue(result, getContext())) {
    resultIsReference = true;
    resultType = value->getType();

    // Failure.
  } else {
    return ConstantEmission();
  }

  // In any case, if the initializer has side-effects, abandon ship.
  if (result.HasSideEffects)
    return ConstantEmission();

  // In CUDA/HIP device compilation, a lambda may capture a reference variable
  // referencing a global host variable by copy. In this case the lambda should
  // make a copy of the value of the global host variable. The DRE of the
  // captured reference variable cannot be emitted as load from the host
  // global variable as compile time constant, since the host variable is not
  // accessible on device. The DRE of the captured reference variable has to be
  // loaded from captures.
  if (CGM.getLangOpts().CUDAIsDevice && result.Val.isLValue() &&
      refExpr->refersToEnclosingVariableOrCapture()) {
    auto *md = dyn_cast_or_null<CXXMethodDecl>(CurCodeDecl);
    if (md && md->getParent()->isLambda() &&
        md->getOverloadedOperator() == OO_Call) {
      const APValue::LValueBase &base = result.Val.getLValueBase();
      if (const ValueDecl *d = base.dyn_cast<const ValueDecl *>()) {
        if (const VarDecl *vd = dyn_cast<const VarDecl>(d)) {
          if (!vd->hasAttr<CUDADeviceAttr>()) {
            return ConstantEmission();
          }
        }
      }
    }
  }

  // Emit as a constant.
  // FIXME(cir): have emitAbstract build a TypedAttr instead (this requires
  // somewhat heavy refactoring...)
  auto c = ConstantEmitter(*this).emitAbstract(refExpr->getLocation(),
                                               result.Val, resultType);
  mlir::TypedAttr cstToEmit = mlir::dyn_cast_if_present<mlir::TypedAttr>(c);
  assert(cstToEmit && "expect a typed attribute");

  // Make sure we emit a debug reference to the global variable.
  // This should probably fire even for
  if (isa<VarDecl>(value)) {
    if (!getContext().DeclMustBeEmitted(cast<VarDecl>(value)))
      buildDeclRefExprDbgValue(refExpr, result.Val);
  } else {
    assert(isa<EnumConstantDecl>(value));
    buildDeclRefExprDbgValue(refExpr, result.Val);
  }

  // If we emitted a reference constant, we need to dereference that.
  if (resultIsReference)
    return ConstantEmission::forReference(cstToEmit);

  return ConstantEmission::forValue(cstToEmit);
}

CIRGenFunction::ConstantEmission
CIRGenFunction::tryEmitAsConstant(const MemberExpr *me) {
  llvm_unreachable("NYI");
}

mlir::Value CIRGenFunction::buildScalarConstant(
    const CIRGenFunction::ConstantEmission &constant, Expr *e) {
  assert(constant && "not a constant");
  if (constant.isReference())
    return buildLoadOfLValue(constant.getReferenceLValue(*this, e),
                             e->getExprLoc())
        .getScalarVal();
  return builder.getConstant(getLoc(e->getSourceRange()), constant.getValue());
}

LValue CIRGenFunction::buildPredefinedLValue(const PredefinedExpr *e) {
  const auto *sl = e->getFunctionName();
  assert(sl != nullptr && "No StringLiteral name in PredefinedExpr");
  auto fn = dyn_cast<mlir::cir::FuncOp>(CurFn);
  assert(fn && "other callables NYI");
  StringRef fnName = fn.getName();
  if (fnName.starts_with("\01"))
    fnName = fnName.substr(1);
  StringRef nameItems[] = {PredefinedExpr::getIdentKindName(e->getIdentKind()),
                           fnName};
  std::string gvName = llvm::join(nameItems, nameItems + 2, ".");
  if (auto *bd = dyn_cast_or_null<BlockDecl>(CurCodeDecl)) {
    llvm_unreachable("NYI");
  }

  return buildStringLiteralLValue(sl);
}
