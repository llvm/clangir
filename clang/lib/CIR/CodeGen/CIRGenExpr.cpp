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

static mlir::cir::FuncOp buildFunctionDeclPointer(CIRGenModule &CGM,
                                                  GlobalDecl GD) {
  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  if (FD->hasAttr<WeakRefAttr>()) {
    mlir::Operation *aliasee = CGM.getWeakRefReference(FD);
    return dyn_cast<FuncOp>(aliasee);
  }

  auto V = CGM.GetAddrOfFunction(GD);

  return V;
}

static Address buildPreserveStructAccess(CIRGenFunction &CGF, LValue base,
                                         Address addr, const FieldDecl *field) {
  llvm_unreachable("NYI");
}

/// Get the address of a zero-sized field within a record. The resulting address
/// doesn't necessarily have the right type.
static Address buildAddrOfFieldStorage(CIRGenFunction &CGF, Address Base,
                                       const FieldDecl *field,
                                       llvm::StringRef fieldName,
                                       unsigned fieldIndex) {
  if (field->isZeroSize(CGF.getContext()))
    llvm_unreachable("NYI");

  auto loc = CGF.getLoc(field->getLocation());

  auto fieldType = CGF.convertType(field->getType());
  auto fieldPtr =
      mlir::cir::PointerType::get(CGF.getBuilder().getContext(), fieldType);
  // For most cases fieldName is the same as field->getName() but for lambdas,
  // which do not currently carry the name, so it can be passed down from the
  // CaptureStmt.
  auto memberAddr = CGF.getBuilder().createGetMember(
      loc, fieldPtr, Base.getPointer(), fieldName, fieldIndex);

  // Retrieve layout information, compute alignment and return the final
  // address.
  const RecordDecl *rec = field->getParent();
  auto &layout = CGF.CGM.getTypes().getCIRGenRecordLayout(rec);
  unsigned idx = layout.getCIRFieldNo(field);
  auto offset = CharUnits::fromQuantity(layout.getCIRType().getElementOffset(
      CGF.CGM.getDataLayout().layout, idx));
  auto addr =
      Address(memberAddr, Base.getAlignment().alignmentAtOffset(offset));
  return addr;
}

static bool hasAnyVptr(const QualType Type, const ASTContext &Context) {
  const auto *RD = Type.getTypePtr()->getAsCXXRecordDecl();
  if (!RD)
    return false;

  if (RD->isDynamicClass())
    return true;

  for (const auto &Base : RD->bases())
    if (hasAnyVptr(Base.getType(), Context))
      return true;

  for (const FieldDecl *Field : RD->fields())
    if (hasAnyVptr(Field->getType(), Context))
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
  if (const CastExpr *CE = dyn_cast<CastExpr>(expr)) {
    if (const auto *ECE = dyn_cast<ExplicitCastExpr>(CE))
      cgf.CGM.buildExplicitCastExprType(ECE, &cgf);

    switch (CE->getCastKind()) {
    // Non-converting casts (but not C's implicit conversion from void*).
    case CK_BitCast:
    case CK_NoOp:
    case CK_AddressSpaceConversion:
      if (auto PtrTy =
              CE->getSubExpr()->getType()->getAs<clang::PointerType>()) {
        if (PtrTy->getPointeeType()->isVoidType())
          break;
        assert(!MissingFeatures::tbaa());

        LValueBaseInfo innerBaseInfo;
        Address addr = cgf.buildPointerWithAlignment(
            CE->getSubExpr(), &innerBaseInfo, tbaaInfo, isKnownNonNull);
        if (baseInfo)
          *baseInfo = innerBaseInfo;

        if (isa<ExplicitCastExpr>(CE)) {
          assert(!MissingFeatures::tbaa());
          LValueBaseInfo TargetTypeBaseInfo;

          CharUnits Align = cgf.CGM.getNaturalPointeeTypeAlignment(
              expr->getType(), &TargetTypeBaseInfo);

          // If the source l-value is opaque, honor the alignment of the
          // casted-to type.
          if (innerBaseInfo.getAlignmentSource() != AlignmentSource::Decl) {
            if (baseInfo)
              baseInfo->mergeForCast(TargetTypeBaseInfo);
            addr = Address(addr.getPointer(), addr.getElementType(), Align,
                           isKnownNonNull);
          }
        }

        if (cgf.SanOpts.has(SanitizerKind::CFIUnrelatedCast) &&
            CE->getCastKind() == CK_BitCast) {
          if (auto PT = expr->getType()->getAs<clang::PointerType>())
            llvm_unreachable("NYI");
        }

        auto ElemTy =
            cgf.getTypes().convertTypeForMem(expr->getType()->getPointeeType());
        addr = cgf.getBuilder().createElementBitCast(
            cgf.getLoc(expr->getSourceRange()), addr, ElemTy);
        if (CE->getCastKind() == CK_AddressSpaceConversion) {
          assert(!MissingFeatures::addressSpace());
          llvm_unreachable("NYI");
        }
        return addr;
      }
      break;

    // Array-to-pointer decay. TODO(cir): BaseInfo and TBAAInfo.
    case CK_ArrayToPointerDecay:
      return cgf.buildArrayToPointerDecay(CE->getSubExpr());

    case CK_UncheckedDerivedToBase:
    case CK_DerivedToBase: {
      // TODO: Support accesses to members of base classes in TBAA. For now, we
      // conservatively pretend that the complete object is of the base class
      // type.
      assert(!MissingFeatures::tbaa());
      Address Addr = cgf.buildPointerWithAlignment(CE->getSubExpr(), baseInfo);
      auto Derived = CE->getSubExpr()->getType()->getPointeeCXXRecordDecl();
      return cgf.getAddressOfBaseClass(
          Addr, Derived, CE->path_begin(), CE->path_end(),
          cgf.shouldNullCheckClassCastValue(CE), CE->getExprLoc());
    }

    // TODO: Is there any reason to treat base-to-derived conversions
    // specially?
    default:
      break;
    }
  }

  // Unary &.
  if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(expr)) {
    // TODO(cir): maybe we should use cir.unary for pointers here instead.
    if (UO->getOpcode() == UO_AddrOf) {
      LValue LV = cgf.buildLValue(UO->getSubExpr());
      if (baseInfo)
        *baseInfo = LV.getBaseInfo();
      assert(!MissingFeatures::tbaa());
      return LV.getAddress();
    }
  }

  // std::addressof and variants.
  if (auto *Call = dyn_cast<CallExpr>(expr)) {
    switch (Call->getBuiltinCallee()) {
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
static bool isAAPCS(const TargetInfo &TargetInfo) {
  return TargetInfo.getABI().starts_with("aapcs");
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

  LValueBaseInfo BaseInfo = base.getBaseInfo();
  const RecordDecl *rec = field->getParent();
  auto &layout = CGM.getTypes().getCIRGenRecordLayout(field->getParent());
  auto &info = layout.getBitFieldInfo(field);
  auto useVolatile = useVolatileForBitField(CGM, base, info, field);
  unsigned Idx = layout.getCIRFieldNo(field);

  if (useVolatile ||
      (IsInPreservedAIRegion ||
       (getDebugInfo() && rec->hasAttr<BPFPreserveAccessIndexAttr>()))) {
    llvm_unreachable("NYI");
  }

  Address Addr = getAddrOfBitFieldStorage(base, field, info.StorageType, Idx);

  auto loc = getLoc(field->getLocation());
  if (Addr.getElementType() != info.StorageType)
    Addr = builder.createElementBitCast(loc, Addr, info.StorageType);

  QualType fieldType =
      field->getType().withCVRQualifiers(base.getVRQualifiers());
  assert(!MissingFeatures::tbaa() && "NYI TBAA for bit fields");
  LValueBaseInfo fieldBaseInfo(BaseInfo.getAlignmentSource());
  return LValue::MakeBitfield(Addr, info, fieldType, fieldBaseInfo,
                              TBAAAccessInfo());
}

LValue CIRGenFunction::buildLValueForField(LValue base,
                                           const FieldDecl *field) {
  LValueBaseInfo BaseInfo = base.getBaseInfo();

  if (field->isBitField())
    return buildLValueForBitField(base, field);

  // Fields of may-alias structures are may-alais themselves.
  // FIXME: this hould get propagated down through anonymous structs and unions.
  QualType FieldType = field->getType();
  const RecordDecl *rec = field->getParent();
  AlignmentSource BaseAlignSource = BaseInfo.getAlignmentSource();
  LValueBaseInfo FieldBaseInfo(getFieldAlignmentSource(BaseAlignSource));
  if (MissingFeatures::tbaa() || rec->hasAttr<MayAliasAttr>() ||
      FieldType->isVectorType()) {
    assert(!MissingFeatures::tbaa() && "NYI");
  } else if (rec->isUnion()) {
    assert(!MissingFeatures::tbaa() && "NYI");
  } else {
    // If no base type been assigned for the base access, then try to generate
    // one for this base lvalue.
    assert(!MissingFeatures::tbaa() && "NYI");
  }

  Address addr = base.getAddress();
  if (auto *ClassDef = dyn_cast<CXXRecordDecl>(rec)) {
    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        ClassDef->isDynamicClass()) {
      llvm_unreachable("NYI");
    }
  }

  unsigned RecordCVR = base.getVRQualifiers();
  if (rec->isUnion()) {
    // NOTE(cir): the element to be loaded/stored need to type-match the
    // source/destination, so we emit a GetMemberOp here.
    llvm::StringRef fieldName = field->getName();
    unsigned fieldIndex = field->getFieldIndex();
    if (CGM.LambdaFieldToName.count(field))
      fieldName = CGM.LambdaFieldToName[field];
    addr = buildAddrOfFieldStorage(*this, addr, field, fieldName, fieldIndex);

    if (CGM.getCodeGenOpts().StrictVTablePointers &&
        hasAnyVptr(FieldType, getContext()))
      // Because unions can easily skip invariant.barriers, we need to add
      // a barrier every time CXXRecord field with vptr is referenced.
      assert(!MissingFeatures::createInvariantGroup());

    if (IsInPreservedAIRegion ||
        (getDebugInfo() && rec->hasAttr<BPFPreserveAccessIndexAttr>())) {
      assert(!MissingFeatures::generateDebugInfo());
    }

    if (FieldType->isReferenceType())
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
  if (FieldType->isReferenceType()) {
    assert(!MissingFeatures::tbaa());
    LValue RefLVal = makeAddrLValue(addr, FieldType, FieldBaseInfo);
    if (RecordCVR & Qualifiers::Volatile)
      RefLVal.getQuals().addVolatile();
    addr = buildLoadOfReference(RefLVal, getLoc(field->getSourceRange()),
                                &FieldBaseInfo);

    // Qualifiers on the struct don't apply to the referencee.
    RecordCVR = 0;
    FieldType = FieldType->getPointeeType();
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
  LValue LV = makeAddrLValue(addr, FieldType, FieldBaseInfo);
  LV.getQuals().addCVRQualifiers(RecordCVR);

  // __weak attribute on a field is ignored.
  if (LV.getQuals().getObjCGCAttr() == Qualifiers::Weak)
    llvm_unreachable("NYI");

  return LV;
}

LValue CIRGenFunction::buildLValueForFieldInitialization(
    LValue Base, const clang::FieldDecl *Field, llvm::StringRef FieldName) {
  QualType FieldType = Field->getType();

  if (!FieldType->isReferenceType())
    return buildLValueForField(Base, Field);

  auto &layout = CGM.getTypes().getCIRGenRecordLayout(Field->getParent());
  unsigned FieldIndex = layout.getCIRFieldNo(Field);

  Address V = buildAddrOfFieldStorage(*this, Base.getAddress(), Field,
                                      FieldName, FieldIndex);

  // Make sure that the address is pointing to the right type.
  auto memTy = getTypes().convertTypeForMem(FieldType);
  V = builder.createElementBitCast(getLoc(Field->getSourceRange()), V, memTy);

  // TODO: Generate TBAA information that describes this access as a structure
  // member access and not just an access to an object of the field's type. This
  // should be similar to what we do in EmitLValueForField().
  LValueBaseInfo BaseInfo = Base.getBaseInfo();
  AlignmentSource FieldAlignSource = BaseInfo.getAlignmentSource();
  LValueBaseInfo FieldBaseInfo(getFieldAlignmentSource(FieldAlignSource));
  assert(!MissingFeatures::tbaa() && "NYI");
  return makeAddrLValue(V, FieldType, FieldBaseInfo);
}

LValue
CIRGenFunction::buildCompoundLiteralLValue(const CompoundLiteralExpr *E) {
  if (E->isFileScope()) {
    llvm_unreachable("NYI");
  }

  if (E->getType()->isVariablyModifiedType()) {
    llvm_unreachable("NYI");
  }

  Address DeclPtr = CreateMemTemp(E->getType(), getLoc(E->getSourceRange()),
                                  ".compoundliteral");
  const Expr *InitExpr = E->getInitializer();
  LValue Result = makeAddrLValue(DeclPtr, E->getType(), AlignmentSource::Decl);

  buildAnyExprToMem(InitExpr, DeclPtr, E->getType().getQualifiers(),
                    /*Init*/ true);

  // Block-scope compound literals are destroyed at the end of the enclosing
  // scope in C.
  if (!getLangOpts().CPlusPlus)
    if (QualType::DestructionKind DtorKind = E->getType().isDestructedType())
      llvm_unreachable("NYI");

  return Result;
}

// Detect the unusual situation where an inline version is shadowed by a
// non-inline version. In that case we should pick the external one
// everywhere. That's GCC behavior too.
static bool onlyHasInlineBuiltinDeclaration(const FunctionDecl *FD) {
  for (const FunctionDecl *PD = FD; PD; PD = PD->getPreviousDecl())
    if (!PD->isInlineBuiltinDeclaration())
      return false;
  return true;
}

static CIRGenCallee buildDirectCallee(CIRGenModule &CGM, GlobalDecl GD) {
  const auto *FD = cast<FunctionDecl>(GD.getDecl());

  if (auto builtinID = FD->getBuiltinID()) {
    std::string NoBuiltinFD = ("no-builtin-" + FD->getName()).str();
    std::string NoBuiltins = "no-builtins";

    auto *A = FD->getAttr<AsmLabelAttr>();
    StringRef Ident = A ? A->getLabel() : FD->getName();
    std::string FDInlineName = (Ident + ".inline").str();

    auto &CGF = *CGM.getCurrCIRGenFun();
    bool IsPredefinedLibFunction =
        CGM.getASTContext().BuiltinInfo.isPredefinedLibFunction(builtinID);
    bool HasAttributeNoBuiltin = false;
    assert(!MissingFeatures::attributeNoBuiltin() && "NYI");
    // bool HasAttributeNoBuiltin =
    //     CGF.CurFn->getAttributes().hasFnAttr(NoBuiltinFD) ||
    //     CGF.CurFn->getAttributes().hasFnAttr(NoBuiltins);

    // When directing calling an inline builtin, call it through it's mangled
    // name to make it clear it's not the actual builtin.
    auto Fn = cast<mlir::cir::FuncOp>(CGF.CurFn);
    if (Fn.getName() != FDInlineName && onlyHasInlineBuiltinDeclaration(FD)) {
      assert(0 && "NYI");
    }

    // Replaceable builtins provide their own implementation of a builtin. If we
    // are in an inline builtin implementation, avoid trivial infinite
    // recursion. Honor __attribute__((no_builtin("foo"))) or
    // __attribute__((no_builtin)) on the current function unless foo is
    // not a predefined library function which means we must generate the
    // builtin no matter what.
    else if (!IsPredefinedLibFunction || !HasAttributeNoBuiltin)
      return CIRGenCallee::forBuiltin(builtinID, FD);
  }

  auto CalleePtr = buildFunctionDeclPointer(CGM, GD);

  assert(!CGM.getLangOpts().CUDA && "NYI");

  return CIRGenCallee::forDirect(CalleePtr, GD);
}

// TODO: this can also be abstrated into common AST helpers
bool CIRGenFunction::hasBooleanRepresentation(QualType Ty) {

  if (Ty->isBooleanType())
    return true;

  if (const EnumType *ET = Ty->getAs<EnumType>())
    return ET->getDecl()->getIntegerType()->isBooleanType();

  if (const AtomicType *AT = Ty->getAs<AtomicType>())
    return hasBooleanRepresentation(AT->getValueType());

  return false;
}

CIRGenCallee CIRGenFunction::buildCallee(const clang::Expr *E) {
  E = E->IgnoreParens();

  // Look through function-to-pointer decay.
  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_FunctionToPointerDecay ||
        ICE->getCastKind() == CK_BuiltinFnToFnPtr) {
      return buildCallee(ICE->getSubExpr());
    }
    // Resolve direct calls.
  } else if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    const auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
    assert(FD &&
           "DeclRef referring to FunctionDecl only thing supported so far");
    return buildDirectCallee(CGM, FD);
  } else if (auto ME = dyn_cast<MemberExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(ME->getMemberDecl())) {
      buildIgnoredExpr(ME->getBase());
      return buildDirectCallee(CGM, FD);
    }
  }

  assert(!dyn_cast<SubstNonTypeTemplateParmExpr>(E) && "NYI");
  assert(!dyn_cast<CXXPseudoDestructorExpr>(E) && "NYI");

  // Otherwise, we have an indirect reference.
  mlir::Value calleePtr;
  QualType functionType;
  if (auto ptrType = E->getType()->getAs<clang::PointerType>()) {
    calleePtr = buildScalarExpr(E);
    functionType = ptrType->getPointeeType();
  } else {
    functionType = E->getType();
    calleePtr = buildLValue(E).getPointer();
  }
  assert(functionType->isFunctionType());

  GlobalDecl GD;
  if (const auto *VD =
          dyn_cast_or_null<VarDecl>(E->getReferencedDeclOfCallee()))
    GD = GlobalDecl(VD);

  CIRGenCalleeInfo calleeInfo(functionType->getAs<FunctionProtoType>(), GD);
  CIRGenCallee callee(calleeInfo, calleePtr.getDefiningOp());
  return callee;

  assert(false && "Nothing else supported yet!");
}

mlir::Value CIRGenFunction::buildToMemory(mlir::Value Value, QualType Ty) {
  // Bool has a different representation in memory than in registers.
  return Value;
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

  mlir::Type SrcTy = value.getType();
  if (const auto *ClangVecTy = ty->getAs<clang::VectorType>()) {
    auto VecTy = dyn_cast<mlir::cir::VectorType>(SrcTy);
    if (!CGM.getCodeGenOpts().PreserveVec3Type &&
        ClangVecTy->getNumElements() == 3) {
      // Handle vec3 special.
      if (VecTy && VecTy.getSize() == 3) {
        // Our source is a vec3, do a shuffle vector to make it a vec4.
        value = builder.createVecShuffle(value.getLoc(), value,
                                         ArrayRef<int64_t>{0, 1, 2, -1});
        SrcTy = mlir::cir::VectorType::get(VecTy.getContext(),
                                           VecTy.getEltType(), 4);
      }
      if (addr.getElementType() != SrcTy) {
        addr = addr.withElementType(SrcTy);
      }
    }
  }

  // Update the alloca with more info on initialization.
  assert(addr.getPointer() && "expected pointer to exist");
  auto SrcAlloca =
      dyn_cast_or_null<mlir::cir::AllocaOp>(addr.getPointer().getDefiningOp());
  if (currVarDecl && SrcAlloca) {
    const VarDecl *VD = currVarDecl;
    assert(VD && "VarDecl expected");
    if (VD->hasInit())
      SrcAlloca.setInitAttr(mlir::UnitAttr::get(builder.getContext()));
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
RValue CIRGenFunction::buildLoadOfLValue(LValue LV, SourceLocation Loc) {
  assert(!LV.getType()->isFunctionType());
  assert(!(LV.getType()->isConstantMatrixType()) && "not implemented");

  if (LV.isBitField())
    return buildLoadOfBitfieldLValue(LV, Loc);

  if (LV.isSimple())
    return RValue::get(buildLoadOfScalar(LV, Loc));

  if (LV.isVectorElt()) {
    auto load = builder.createLoad(getLoc(Loc), LV.getVectorAddress());
    return RValue::get(builder.create<mlir::cir::VecExtractOp>(
        getLoc(Loc), load, LV.getVectorIdx()));
  }

  if (LV.isExtVectorElt()) {
    return buildLoadOfExtVectorElementLValue(LV);
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
RValue CIRGenFunction::buildLoadOfExtVectorElementLValue(LValue LV) {
  mlir::Location loc = LV.getExtVectorPointer().getLoc();
  mlir::Value Vec = builder.createLoad(loc, LV.getExtVectorAddress());

  // HLSL allows treating scalars as one-element vectors. Converting the scalar
  // IR value to a vector here allows the rest of codegen to behave as normal.
  if (getLangOpts().HLSL && !mlir::isa<mlir::cir::VectorType>(Vec.getType())) {
    llvm_unreachable("HLSL NYI");
  }

  const mlir::ArrayAttr Elts = LV.getExtVectorElts();

  // If the result of the expression is a non-vector type, we must be extracting
  // a single element.  Just codegen as an extractelement.
  const auto *ExprVT = LV.getType()->getAs<clang::VectorType>();
  if (!ExprVT) {
    int64_t InIdx = getAccessedFieldNo(0, Elts);
    mlir::cir::ConstantOp Elt =
        builder.getConstInt(loc, builder.getSInt64Ty(), InIdx);
    return RValue::get(builder.create<mlir::cir::VecExtractOp>(loc, Vec, Elt));
  }

  // Always use shuffle vector to try to retain the original program structure
  unsigned NumResultElts = ExprVT->getNumElements();

  SmallVector<int64_t, 4> Mask;
  for (unsigned i = 0; i != NumResultElts; ++i)
    Mask.push_back(getAccessedFieldNo(i, Elts));

  Vec = builder.createVecShuffle(loc, Vec, Mask);
  return RValue::get(Vec);
}

RValue CIRGenFunction::buildLoadOfBitfieldLValue(LValue LV,
                                                 SourceLocation Loc) {
  const CIRGenBitFieldInfo &info = LV.getBitFieldInfo();

  // Get the output type.
  mlir::Type resLTy = convertType(LV.getType());
  Address ptr = LV.getBitFieldAddress();

  bool useVolatile = LV.isVolatileQualified() &&
                     info.VolatileStorageSize != 0 && isAAPCS(CGM.getTarget());

  auto field = builder.createGetBitfield(getLoc(Loc), resLTy, ptr.getPointer(),
                                         ptr.getElementType(), info,
                                         LV.isVolatile(), useVolatile);
  assert(!MissingFeatures::emitScalarRangeCheck() && "NYI");
  return RValue::get(field);
}

void CIRGenFunction::buildStoreThroughExtVectorComponentLValue(RValue Src,
                                                               LValue Dst) {
  mlir::Location loc = Dst.getExtVectorPointer().getLoc();

  // HLSL allows storing to scalar values through ExtVector component LValues.
  // To support this we need to handle the case where the destination address is
  // a scalar.
  Address DstAddr = Dst.getExtVectorAddress();
  if (!mlir::isa<mlir::cir::VectorType>(DstAddr.getElementType())) {
    llvm_unreachable("HLSL NYI");
  }

  // This access turns into a read/modify/write of the vector.  Load the input
  // value now.
  mlir::Value Vec = builder.createLoad(loc, DstAddr);
  const mlir::ArrayAttr Elts = Dst.getExtVectorElts();

  mlir::Value SrcVal = Src.getScalarVal();

  if (const clang::VectorType *VTy =
          Dst.getType()->getAs<clang::VectorType>()) {
    unsigned NumSrcElts = VTy->getNumElements();
    unsigned NumDstElts = cast<mlir::cir::VectorType>(Vec.getType()).getSize();
    if (NumDstElts == NumSrcElts) {
      // Use shuffle vector is the src and destination are the same number of
      // elements and restore the vector mask since it is on the side it will be
      // stored.
      SmallVector<int64_t, 4> Mask(NumDstElts);
      for (unsigned i = 0; i != NumSrcElts; ++i)
        Mask[getAccessedFieldNo(i, Elts)] = i;

      Vec = builder.createVecShuffle(loc, SrcVal, Mask);
    } else if (NumDstElts > NumSrcElts) {
      // Extended the source vector to the same length and then shuffle it
      // into the destination.
      // FIXME: since we're shuffling with undef, can we just use the indices
      //        into that?  This could be simpler.
      SmallVector<int64_t, 4> ExtMask;
      for (unsigned i = 0; i != NumSrcElts; ++i)
        ExtMask.push_back(i);
      ExtMask.resize(NumDstElts, -1);
      mlir::Value ExtSrcVal = builder.createVecShuffle(loc, SrcVal, ExtMask);
      // build identity
      SmallVector<int64_t, 4> Mask;
      for (unsigned i = 0; i != NumDstElts; ++i)
        Mask.push_back(i);

      // When the vector size is odd and .odd or .hi is used, the last element
      // of the Elts constant array will be one past the size of the vector.
      // Ignore the last element here, if it is greater than the mask size.
      if ((unsigned)getAccessedFieldNo(NumSrcElts - 1, Elts) == Mask.size())
        NumSrcElts--;

      // modify when what gets shuffled in
      for (unsigned i = 0; i != NumSrcElts; ++i)
        Mask[getAccessedFieldNo(i, Elts)] = i + NumDstElts;
      Vec = builder.createVecShuffle(loc, Vec, ExtSrcVal, Mask);
    } else {
      // We should never shorten the vector
      llvm_unreachable("unexpected shorten vector length");
    }
  } else {
    // If the Src is a scalar (not a vector), and the target is a vector it must
    // be updating one element.
    unsigned InIdx = getAccessedFieldNo(0, Elts);
    auto Elt = builder.getSInt64(InIdx, loc);

    Vec = builder.create<mlir::cir::VecInsertOp>(loc, Vec, SrcVal, Elt);
  }

  builder.createStore(loc, Vec, Dst.getExtVectorAddress(),
                      Dst.isVolatileQualified());
}

void CIRGenFunction::buildStoreThroughLValue(RValue Src, LValue Dst,
                                             bool isInit) {
  if (!Dst.isSimple()) {
    if (Dst.isVectorElt()) {
      // Read/modify/write the vector, inserting the new element
      mlir::Location loc = Dst.getVectorPointer().getLoc();
      mlir::Value Vector = builder.createLoad(loc, Dst.getVectorAddress());
      Vector = builder.create<mlir::cir::VecInsertOp>(
          loc, Vector, Src.getScalarVal(), Dst.getVectorIdx());
      builder.createStore(loc, Vector, Dst.getVectorAddress());
      return;
    }

    if (Dst.isExtVectorElt())
      return buildStoreThroughExtVectorComponentLValue(Src, Dst);

    assert(Dst.isBitField() && "NIY LValue type");
    mlir::Value result;
    return buildStoreThroughBitfieldLValue(Src, Dst, result);
  }
  assert(Dst.isSimple() && "only implemented simple");

  // There's special magic for assigning into an ARC-qualified l-value.
  if (Qualifiers::ObjCLifetime Lifetime = Dst.getQuals().getObjCLifetime()) {
    llvm_unreachable("NYI");
  }

  if (Dst.isObjCWeak() && !Dst.isNonGC()) {
    llvm_unreachable("NYI");
  }

  if (Dst.isObjCStrong() && !Dst.isNonGC()) {
    llvm_unreachable("NYI");
  }

  assert(Src.isScalar() && "Can't emit an agg store with this method");
  buildStoreOfScalar(Src.getScalarVal(), Dst, isInit);
}

void CIRGenFunction::buildStoreThroughBitfieldLValue(RValue Src, LValue Dst,
                                                     mlir::Value &Result) {
  // According to the AACPS:
  // When a volatile bit-field is written, and its container does not overlap
  // with any non-bit-field member, its container must be read exactly once
  // and written exactly once using the access width appropriate to the type
  // of the container. The two accesses are not atomic.
  if (Dst.isVolatileQualified() && isAAPCS(CGM.getTarget()) &&
      CGM.getCodeGenOpts().ForceAAPCSBitfieldLoad)
    llvm_unreachable("volatile bit-field is not implemented for the AACPS");

  const CIRGenBitFieldInfo &info = Dst.getBitFieldInfo();
  mlir::Type resLTy = getTypes().convertTypeForMem(Dst.getType());
  Address ptr = Dst.getBitFieldAddress();

  const bool useVolatile =
      CGM.getCodeGenOpts().AAPCSBitfieldWidth && Dst.isVolatileQualified() &&
      info.VolatileStorageSize != 0 && isAAPCS(CGM.getTarget());

  mlir::Value dstAddr = Dst.getAddress().getPointer();

  Result = builder.createSetBitfield(
      dstAddr.getLoc(), resLTy, dstAddr, ptr.getElementType(),
      Src.getScalarVal(), info, Dst.isVolatileQualified(), useVolatile);
}

static LValue buildGlobalVarDeclLValue(CIRGenFunction &CGF, const Expr *E,
                                       const VarDecl *VD) {
  QualType T = E->getType();

  // If it's thread_local, emit a call to its wrapper function instead.
  if (VD->getTLSKind() == VarDecl::TLS_Dynamic &&
      CGF.CGM.getCXXABI().usesThreadWrapperFunction(VD))
    assert(0 && "not implemented");

  // Check if the variable is marked as declare target with link clause in
  // device codegen.
  if (CGF.getLangOpts().OpenMP)
    llvm_unreachable("not implemented");

  // Traditional LLVM codegen handles thread local separately, CIR handles
  // as part of getAddrOfGlobalVar.
  auto V = CGF.CGM.getAddrOfGlobalVar(VD);

  auto RealVarTy = CGF.getTypes().convertTypeForMem(VD->getType());
  mlir::cir::PointerType realPtrTy = CGF.getBuilder().getPointerTo(
      RealVarTy, cast_if_present<mlir::cir::AddressSpaceAttr>(
                     cast<mlir::cir::PointerType>(V.getType()).getAddrSpace()));
  if (realPtrTy != V.getType())
    V = CGF.getBuilder().createBitcast(V.getLoc(), V, realPtrTy);

  CharUnits Alignment = CGF.getContext().getDeclAlign(VD);
  Address Addr(V, RealVarTy, Alignment);
  // Emit reference to the private copy of the variable if it is an OpenMP
  // threadprivate variable.
  if (CGF.getLangOpts().OpenMP && !CGF.getLangOpts().OpenMPSimd &&
      VD->hasAttr<clang::OMPThreadPrivateDeclAttr>()) {
    assert(0 && "NYI");
  }
  LValue LV;
  if (VD->getType()->isReferenceType())
    assert(0 && "NYI");
  else
    LV = CGF.makeAddrLValue(Addr, T, AlignmentSource::Decl);
  assert(!MissingFeatures::setObjCGCLValueClass() && "NYI");
  return LV;
}

static LValue buildCapturedFieldLValue(CIRGenFunction &CGF, const FieldDecl *FD,
                                       mlir::Value ThisValue) {
  QualType TagType = CGF.getContext().getTagDeclType(FD->getParent());
  LValue LV = CGF.MakeNaturalAlignAddrLValue(ThisValue, TagType);
  return CGF.buildLValueForField(LV, FD);
}

static LValue buildFunctionDeclLValue(CIRGenFunction &CGF, const Expr *E,
                                      GlobalDecl GD) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
  auto funcOp = buildFunctionDeclPointer(CGF.CGM, GD);
  auto loc = CGF.getLoc(E->getSourceRange());
  CharUnits align = CGF.getContext().getDeclAlign(FD);

  mlir::Type fnTy = funcOp.getFunctionType();
  auto ptrTy = mlir::cir::PointerType::get(CGF.getBuilder().getContext(), fnTy);
  mlir::Value addr = CGF.getBuilder().create<mlir::cir::GetGlobalOp>(
      loc, ptrTy, funcOp.getSymName());

  if (funcOp.getFunctionType() !=
      CGF.CGM.getTypes().ConvertType(FD->getType())) {
    fnTy = CGF.CGM.getTypes().ConvertType(FD->getType());
    ptrTy = mlir::cir::PointerType::get(CGF.getBuilder().getContext(), fnTy);

    addr = CGF.getBuilder().create<mlir::cir::CastOp>(
        addr.getLoc(), ptrTy, mlir::cir::CastKind::bitcast, addr);
  }

  return CGF.makeAddrLValue(Address(addr, fnTy, align), E->getType(),
                            AlignmentSource::Decl);
}

LValue CIRGenFunction::buildDeclRefLValue(const DeclRefExpr *E) {
  const NamedDecl *ND = E->getDecl();
  QualType T = E->getType();

  assert(E->isNonOdrUse() != NOUR_Unevaluated &&
         "should not emit an unevaluated operand");

  if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    // Global Named registers access via intrinsics only
    if (VD->getStorageClass() == SC_Register && VD->hasAttr<AsmLabelAttr>() &&
        !VD->isLocalVarDecl())
      llvm_unreachable("NYI");

    assert(E->isNonOdrUse() != NOUR_Constant && "not implemented");

    // Check for captured variables.
    if (E->refersToEnclosingVariableOrCapture()) {
      VD = VD->getCanonicalDecl();
      if (auto *FD = LambdaCaptureFields.lookup(VD))
        return buildCapturedFieldLValue(*this, FD, CXXABIThisValue);
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
  assert((ND->isUsed(false) || !isa<VarDecl>(ND) || E->isNonOdrUse() ||
          !E->getLocation().isValid()) &&
         "Should not use decl without marking it used!");

  if (ND->hasAttr<WeakRefAttr>()) {
    llvm_unreachable("NYI");
  }

  if (const auto *VD = dyn_cast<VarDecl>(ND)) {
    // Check if this is a global variable
    if (VD->hasLinkage() || VD->isStaticDataMember())
      return buildGlobalVarDeclLValue(*this, E, VD);

    Address addr = Address::invalid();

    // The variable should generally be present in the local decl map.
    auto iter = LocalDeclMap.find(VD);
    if (iter != LocalDeclMap.end()) {
      addr = iter->second;
    }
    // Otherwise, it might be static local we haven't emitted yet for some
    // reason; most likely, because it's in an outer function.
    else if (VD->isStaticLocal()) {
      mlir::cir::GlobalOp var = CGM.getOrCreateStaticVarDecl(
          *VD, CGM.getCIRLinkageVarDefinition(VD, /*IsConstant=*/false));
      addr = Address(builder.createGetGlobal(var), convertType(VD->getType()),
                     getContext().getDeclAlign(VD));
    } else {
      llvm_unreachable("DeclRefExpr for decl not entered in LocalDeclMap?");
    }

    // Handle threadlocal function locals.
    if (VD->getTLSKind() != VarDecl::TLS_None)
      llvm_unreachable("thread-local storage is NYI");

    // Check for OpenMP threadprivate variables.
    if (getLangOpts().OpenMP && !getLangOpts().OpenMPSimd &&
        VD->hasAttr<OMPThreadPrivateDeclAttr>()) {
      llvm_unreachable("NYI");
    }

    // Drill into block byref variables.
    bool isBlockByref = VD->isEscapingByref();
    if (isBlockByref) {
      llvm_unreachable("NYI");
    }

    // Drill into reference types.
    LValue LV =
        VD->getType()->isReferenceType()
            ? buildLoadOfReferenceLValue(addr, getLoc(E->getSourceRange()),
                                         VD->getType(), AlignmentSource::Decl)
            : makeAddrLValue(addr, T, AlignmentSource::Decl);

    // Statics are defined as globals, so they are not include in the function's
    // symbol table.
    assert((VD->isStaticLocal() || symbolTable.count(VD)) &&
           "non-static locals should be already mapped");

    bool isLocalStorage = VD->hasLocalStorage();

    bool NonGCable =
        isLocalStorage && !VD->getType()->isReferenceType() && !isBlockByref;

    if (NonGCable && MissingFeatures::setNonGC()) {
      llvm_unreachable("garbage collection is NYI");
    }

    bool isImpreciseLifetime =
        (isLocalStorage && !VD->hasAttr<ObjCPreciseLifetimeAttr>());
    if (isImpreciseLifetime && MissingFeatures::ARC())
      llvm_unreachable("imprecise lifetime is NYI");
    assert(!MissingFeatures::setObjCGCLValueClass());

    // Statics are defined as globals, so they are not include in the function's
    // symbol table.
    assert((VD->isStaticLocal() || symbolTable.lookup(VD)) &&
           "Name lookup must succeed for non-static local variables");

    return LV;
  }

  if (const auto *FD = dyn_cast<FunctionDecl>(ND)) {
    LValue LV = buildFunctionDeclLValue(*this, E, FD);

    // Emit debuginfo for the function declaration if the target wants to.
    if (getContext().getTargetInfo().allowDebugInfoForExternalRef())
      assert(!MissingFeatures::generateDebugInfo());

    return LV;
  }

  // FIXME: While we're emitting a binding from an enclosing scope, all other
  // DeclRefExprs we see should be implicitly treated as if they also refer to
  // an enclosing scope.
  if (const auto *BD = dyn_cast<BindingDecl>(ND)) {
    if (E->refersToEnclosingVariableOrCapture()) {
      auto *FD = LambdaCaptureFields.lookup(BD);
      return buildCapturedFieldLValue(*this, FD, CXXABIThisValue);
    }
    return buildLValue(BD->getBinding());
  }

  // We can form DeclRefExprs naming GUID declarations when reconstituting
  // non-type template parameters into expressions.
  if (const auto *GD = dyn_cast<MSGuidDecl>(ND))
    llvm_unreachable("NYI");

  if (const auto *TPO = dyn_cast<TemplateParamObjectDecl>(ND))
    llvm_unreachable("NYI");

  llvm_unreachable("Unhandled DeclRefExpr");
}

LValue
CIRGenFunction::buildPointerToDataMemberBinaryExpr(const BinaryOperator *E) {
  assert((E->getOpcode() == BO_PtrMemD || E->getOpcode() == BO_PtrMemI) &&
         "unexpected binary operator opcode");

  auto baseAddr = Address::invalid();
  if (E->getOpcode() == BO_PtrMemD)
    baseAddr = buildLValue(E->getLHS()).getAddress();
  else
    baseAddr = buildPointerWithAlignment(E->getLHS());

  const auto *memberPtrTy = E->getRHS()->getType()->castAs<MemberPointerType>();

  auto memberPtr = buildScalarExpr(E->getRHS());

  LValueBaseInfo baseInfo;
  // TODO(cir): add TBAA
  assert(!MissingFeatures::tbaa());
  auto memberAddr = buildCXXMemberDataPointerAddress(E, baseAddr, memberPtr,
                                                     memberPtrTy, &baseInfo);

  return makeAddrLValue(memberAddr, memberPtrTy->getPointeeType(), baseInfo);
}

LValue
CIRGenFunction::buildExtVectorElementExpr(const ExtVectorElementExpr *E) {
  // Emit the base vector as an l-value.
  LValue base;

  // ExtVectorElementExpr's base can either be a vector or pointer to vector.
  if (E->isArrow()) {
    // If it is a pointer to a vector, emit the address and form an lvalue with
    // it.
    LValueBaseInfo BaseInfo;
    // TODO(cir): Support TBAA
    assert(!MissingFeatures::tbaa());
    Address Ptr = buildPointerWithAlignment(E->getBase(), &BaseInfo);
    const auto *PT = E->getBase()->getType()->castAs<clang::PointerType>();
    base = makeAddrLValue(Ptr, PT->getPointeeType(), BaseInfo);
    base.getQuals().removeObjCGCAttr();
  } else if (E->getBase()->isGLValue()) {
    // Otherwise, if the base is an lvalue ( as in the case of foo.x.x),
    // emit the base as an lvalue.
    assert(E->getBase()->getType()->isVectorType());
    base = buildLValue(E->getBase());
  } else {
    // Otherwise, the base is a normal rvalue (as in (V+V).x), emit it as such.
    assert(E->getBase()->getType()->isVectorType() &&
           "Result must be a vector");
    mlir::Value Vec = buildScalarExpr(E->getBase());

    // Store the vector to memory (because LValue wants an address).
    QualType BaseTy = E->getBase()->getType();
    Address VecMem = CreateMemTemp(BaseTy, Vec.getLoc(), "tmp");
    builder.createStore(Vec.getLoc(), Vec, VecMem);
    base = makeAddrLValue(VecMem, BaseTy, AlignmentSource::Decl);
  }

  QualType type =
      E->getType().withCVRQualifiers(base.getQuals().getCVRQualifiers());

  // Encode the element access list into a vector of unsigned indices.
  SmallVector<uint32_t, 4> indices;
  E->getEncodedElementAccess(indices);

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

LValue CIRGenFunction::buildBinaryOperatorLValue(const BinaryOperator *E) {
  // Comma expressions just emit their LHS then their RHS as an l-value.
  if (E->getOpcode() == BO_Comma) {
    buildIgnoredExpr(E->getLHS());
    return buildLValue(E->getRHS());
  }

  if (E->getOpcode() == BO_PtrMemD || E->getOpcode() == BO_PtrMemI)
    return buildPointerToDataMemberBinaryExpr(E);

  assert(E->getOpcode() == BO_Assign && "unexpected binary l-value");

  // Note that in all of these cases, __block variables need the RHS
  // evaluated first just in case the variable gets moved by the RHS.

  switch (CIRGenFunction::getEvaluationKind(E->getType())) {
  case TEK_Scalar: {
    assert(E->getLHS()->getType().getObjCLifetime() ==
               clang::Qualifiers::ObjCLifetime::OCL_None &&
           "not implemented");

    RValue RV = buildAnyExpr(E->getRHS());
    LValue LV = buildLValue(E->getLHS());

    SourceLocRAIIObject Loc{*this, getLoc(E->getSourceRange())};
    if (LV.isBitField()) {
      mlir::Value result;
      buildStoreThroughBitfieldLValue(RV, LV, result);
    } else {
      buildStoreThroughLValue(RV, LV);
    }
    if (getLangOpts().OpenMP)
      CGM.getOpenMPRuntime().checkAndEmitLastprivateConditional(*this,
                                                                E->getLHS());
    return LV;
  }

  case TEK_Complex:
    return buildComplexAssignmentLValue(E);
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
mlir::Value CIRGenFunction::evaluateExprAsBool(const Expr *E) {
  // TODO(cir): PGO
  if (const MemberPointerType *MPT = E->getType()->getAs<MemberPointerType>()) {
    assert(0 && "not implemented");
  }

  QualType BoolTy = getContext().BoolTy;
  SourceLocation Loc = E->getExprLoc();
  // TODO(cir): CGFPOptionsRAII for FP stuff.
  if (!E->getType()->isAnyComplexType())
    return buildScalarConversion(buildScalarExpr(E), E->getType(), BoolTy, Loc);

  llvm_unreachable("complex to scalar not implemented");
}

LValue CIRGenFunction::buildUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UO_Extension)
    return buildLValue(E->getSubExpr());

  QualType ExprTy = getContext().getCanonicalType(E->getSubExpr()->getType());
  switch (E->getOpcode()) {
  default:
    llvm_unreachable("Unknown unary operator lvalue!");
  case UO_Deref: {
    QualType T = E->getSubExpr()->getType()->getPointeeType();
    assert(!T.isNull() && "CodeGenFunction::EmitUnaryOpLValue: Illegal type");

    LValueBaseInfo BaseInfo;
    // TODO: add TBAAInfo
    Address Addr = buildPointerWithAlignment(E->getSubExpr(), &BaseInfo);

    // Tag 'load' with deref attribute.
    if (auto loadOp =
            dyn_cast<::mlir::cir::LoadOp>(Addr.getPointer().getDefiningOp())) {
      loadOp.setIsDerefAttr(mlir::UnitAttr::get(builder.getContext()));
    }

    LValue LV = LValue::makeAddr(Addr, T, BaseInfo);
    // TODO: set addr space
    // TODO: ObjC/GC/__weak write barrier stuff.
    return LV;
  }
  case UO_Real:
  case UO_Imag: {
    LValue LV = buildLValue(E->getSubExpr());
    assert(LV.isSimple() && "real/imag on non-ordinary l-value");

    // __real is valid on scalars.  This is a faster way of testing that.
    // __imag can only produce an rvalue on scalars.
    if (E->getOpcode() == UO_Real &&
        !mlir::isa<mlir::cir::ComplexType>(LV.getAddress().getElementType())) {
      assert(E->getSubExpr()->getType()->isArithmeticType());
      return LV;
    }

    QualType T = ExprTy->castAs<clang::ComplexType>()->getElementType();

    auto Loc = getLoc(E->getExprLoc());
    Address Component =
        (E->getOpcode() == UO_Real
             ? buildAddrOfRealComponent(Loc, LV.getAddress(), LV.getType())
             : buildAddrOfImagComponent(Loc, LV.getAddress(), LV.getType()));
    // TODO(cir): TBAA info.
    assert(!MissingFeatures::tbaa());
    LValue ElemLV = makeAddrLValue(Component, T, LV.getBaseInfo());
    ElemLV.getQuals().addQualifiers(LV.getQuals());
    return ElemLV;
  }
  case UO_PreInc:
  case UO_PreDec: {
    bool isInc = E->isIncrementOp();
    bool isPre = E->isPrefix();
    LValue LV = buildLValue(E->getSubExpr());

    if (E->getType()->isAnyComplexType()) {
      buildComplexPrePostIncDec(E, LV, isInc, true /*isPre*/);
    } else {
      buildScalarPrePostIncDec(E, LV, isInc, isPre);
    }

    return LV;
  }
  }
}

/// Emit code to compute the specified expression which
/// can have any type.  The result is returned as an RValue struct.
RValue CIRGenFunction::buildAnyExpr(const Expr *E, AggValueSlot aggSlot,
                                    bool ignoreResult) {
  switch (CIRGenFunction::getEvaluationKind(E->getType())) {
  case TEK_Scalar:
    return RValue::get(buildScalarExpr(E));
  case TEK_Complex:
    return RValue::getComplex(buildComplexExpr(E));
  case TEK_Aggregate: {
    if (!ignoreResult && aggSlot.isIgnored())
      aggSlot = CreateAggTemp(E->getType(), getLoc(E->getSourceRange()),
                              getCounterAggTmpAsString());
    buildAggExpr(E, aggSlot);
    return aggSlot.asRValue();
  }
  }
  llvm_unreachable("bad evaluation kind");
}

RValue CIRGenFunction::buildCallExpr(const clang::CallExpr *E,
                                     ReturnValueSlot ReturnValue) {
  assert(!E->getCallee()->getType()->isBlockPointerType() && "ObjC Blocks NYI");

  if (const auto *CE = dyn_cast<CXXMemberCallExpr>(E))
    return buildCXXMemberCallExpr(CE, ReturnValue);

  assert(!dyn_cast<CUDAKernelCallExpr>(E) && "CUDA NYI");
  if (const auto *CE = dyn_cast<CXXOperatorCallExpr>(E))
    if (const CXXMethodDecl *MD =
            dyn_cast_or_null<CXXMethodDecl>(CE->getCalleeDecl()))
      return buildCXXOperatorMemberCallExpr(CE, MD, ReturnValue);

  CIRGenCallee callee = buildCallee(E->getCallee());

  if (callee.isBuiltin())
    return buildBuiltinExpr(callee.getBuiltinDecl(), callee.getBuiltinID(), E,
                            ReturnValue);

  assert(!callee.isPsuedoDestructor() && "NYI");

  return buildCall(E->getCallee()->getType(), callee, E, ReturnValue);
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

LValue CIRGenFunction::buildStmtExprLValue(const StmtExpr *E) {
  // Can only get l-value for message expression returning aggregate type
  RValue RV = buildAnyExprToTemp(E);
  return makeAddrLValue(RV.getAggregateAddress(), E->getType(),
                        AlignmentSource::Decl);
}

RValue CIRGenFunction::buildCall(clang::QualType CalleeType,
                                 const CIRGenCallee &OrigCallee,
                                 const clang::CallExpr *E,
                                 ReturnValueSlot ReturnValue,
                                 mlir::Value Chain) {
  // Get the actual function type. The callee type will always be a pointer to
  // function type or a block pointer type.
  assert(CalleeType->isFunctionPointerType() &&
         "Call must have function pointer type!");

  auto *TargetDecl = OrigCallee.getAbstractInfo().getCalleeDecl().getDecl();
  (void)TargetDecl;

  CalleeType = getContext().getCanonicalType(CalleeType);

  auto PointeeType = cast<clang::PointerType>(CalleeType)->getPointeeType();

  CIRGenCallee Callee = OrigCallee;

  if (getLangOpts().CPlusPlus)
    assert(!SanOpts.has(SanitizerKind::Function) && "Sanitizers NYI");

  const auto *FnType = cast<FunctionType>(PointeeType);

  assert(!SanOpts.has(SanitizerKind::CFIICall) && "Sanitizers NYI");

  CallArgList Args;

  assert(!Chain && "FIX THIS");

  // C++17 requires that we evaluate arguments to a call using assignment syntax
  // right-to-left, and that we evaluate arguments to certain other operators
  // left-to-right. Note that we allow this to override the order dictated by
  // the calling convention on the MS ABI, which means that parameter
  // destruction order is not necessarily reverse construction order.
  // FIXME: Revisit this based on C++ committee response to unimplementability.
  EvaluationOrder Order = EvaluationOrder::Default;
  if (auto *OCE = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (OCE->isAssignmentOp())
      Order = EvaluationOrder::ForceRightToLeft;
    else {
      switch (OCE->getOperator()) {
      case OO_LessLess:
      case OO_GreaterGreater:
      case OO_AmpAmp:
      case OO_PipePipe:
      case OO_Comma:
      case OO_ArrowStar:
        Order = EvaluationOrder::ForceLeftToRight;
        break;
      default:
        break;
      }
    }
  }

  buildCallArgs(Args, dyn_cast<FunctionProtoType>(FnType), E->arguments(),
                E->getDirectCallee(), /*ParamsToSkip*/ 0, Order);

  const CIRGenFunctionInfo &FnInfo = CGM.getTypes().arrangeFreeFunctionCall(
      Args, FnType, /*ChainCall=*/Chain.getAsOpaquePointer());

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
  if (isa<FunctionNoProtoType>(FnType) || Chain) {
    assert(!MissingFeatures::chainCall());
    assert(!MissingFeatures::addressSpace());
    auto CalleeTy = getTypes().GetFunctionType(FnInfo);
    // get non-variadic function type
    CalleeTy = mlir::cir::FuncType::get(CalleeTy.getInputs(),
                                        CalleeTy.getReturnType(), false);
    auto CalleePtrTy =
        mlir::cir::PointerType::get(builder.getContext(), CalleeTy);

    auto *Fn = Callee.getFunctionPointer();
    mlir::Value Addr;
    if (auto funcOp = llvm::dyn_cast<mlir::cir::FuncOp>(Fn)) {
      Addr = builder.create<mlir::cir::GetGlobalOp>(
          getLoc(E->getSourceRange()),
          mlir::cir::PointerType::get(builder.getContext(),
                                      funcOp.getFunctionType()),
          funcOp.getSymName());
    } else {
      Addr = Fn->getResult(0);
    }

    Fn = builder.createBitcast(Addr, CalleePtrTy).getDefiningOp();
    Callee.setFunctionPointer(Fn);
  }

  assert(!CGM.getLangOpts().HIP && "HIP NYI");

  assert(!MustTailCall && "Must tail NYI");
  mlir::cir::CIRCallOpInterface callOP;
  RValue Call = buildCall(FnInfo, Callee, ReturnValue, Args, &callOP,
                          E == MustTailCall, getLoc(E->getExprLoc()), E);

  assert(!getDebugInfo() && "Debug Info NYI");

  return Call;
}

/// Emit code to compute the specified expression, ignoring the result.
void CIRGenFunction::buildIgnoredExpr(const Expr *E) {
  if (E->isPRValue())
    return (void)buildAnyExpr(E, AggValueSlot::ignored(), true);

  // Just emit it as an l-value and drop the result.
  buildLValue(E);
}

Address CIRGenFunction::buildArrayToPointerDecay(const Expr *E,
                                                 LValueBaseInfo *BaseInfo) {
  assert(E->getType()->isArrayType() &&
         "Array to pointer decay must have array source type!");

  // Expressions of array type can't be bitfields or vector elements.
  LValue LV = buildLValue(E);
  Address Addr = LV.getAddress();

  // If the array type was an incomplete type, we need to make sure
  // the decay ends up being the right type.
  auto lvalueAddrTy =
      mlir::dyn_cast<mlir::cir::PointerType>(Addr.getPointer().getType());
  assert(lvalueAddrTy && "expected pointer");

  if (E->getType()->isVariableArrayType())
    return Addr;

  auto pointeeTy =
      mlir::dyn_cast<mlir::cir::ArrayType>(lvalueAddrTy.getPointee());
  assert(pointeeTy && "expected array");

  mlir::Type arrayTy = convertType(E->getType());
  assert(mlir::isa<mlir::cir::ArrayType>(arrayTy) && "expected array");
  assert(pointeeTy == arrayTy);

  // The result of this decay conversion points to an array element within the
  // base lvalue. However, since TBAA currently does not support representing
  // accesses to elements of member arrays, we conservatively represent accesses
  // to the pointee object as if it had no any base lvalue specified.
  // TODO: Support TBAA for member arrays.
  QualType EltType = E->getType()->castAsArrayTypeUnsafe()->getElementType();
  if (BaseInfo)
    *BaseInfo = LV.getBaseInfo();
  assert(!MissingFeatures::tbaa() && "NYI");

  mlir::Value ptr = CGM.getBuilder().maybeBuildArrayDecay(
      CGM.getLoc(E->getSourceRange()), Addr.getPointer(),
      getTypes().convertTypeForMem(EltType));
  return Address(ptr, Addr.getAlignment());
}

/// If the specified expr is a simple decay from an array to pointer,
/// return the array subexpression.
/// FIXME: this could be abstracted into a commeon AST helper.
static const Expr *isSimpleArrayDecayOperand(const Expr *E) {
  // If this isn't just an array->pointer decay, bail out.
  const auto *CE = dyn_cast<CastExpr>(E);
  if (!CE || CE->getCastKind() != CK_ArrayToPointerDecay)
    return nullptr;

  // If this is a decay from variable width array, bail out.
  const Expr *SubExpr = CE->getSubExpr();
  if (SubExpr->getType()->isVariableArrayType())
    return nullptr;

  return SubExpr;
}

/// Given an array base, check whether its member access belongs to a record
/// with preserve_access_index attribute or not.
/// TODO(cir): don't need to be specific to LLVM's codegen, refactor into common
/// AST helpers.
static bool isPreserveAIArrayBase(CIRGenFunction &CGF, const Expr *ArrayBase) {
  if (!ArrayBase || !CGF.getDebugInfo())
    return false;

  // Only support base as either a MemberExpr or DeclRefExpr.
  // DeclRefExpr to cover cases like:
  //    struct s { int a; int b[10]; };
  //    struct s *p;
  //    p[1].a
  // p[1] will generate a DeclRefExpr and p[1].a is a MemberExpr.
  // p->b[5] is a MemberExpr example.
  const Expr *E = ArrayBase->IgnoreImpCasts();
  if (const auto *ME = dyn_cast<MemberExpr>(E))
    return ME->getMemberDecl()->hasAttr<BPFPreserveAccessIndexAttr>();

  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    const auto *VarDef = dyn_cast<VarDecl>(DRE->getDecl());
    if (!VarDef)
      return false;

    const auto *PtrT = VarDef->getType()->getAs<clang::PointerType>();
    if (!PtrT)
      return false;

    const auto *PointeeT =
        PtrT->getPointeeType()->getUnqualifiedDesugaredType();
    if (const auto *RecT = dyn_cast<RecordType>(PointeeT))
      return RecT->getDecl()->hasAttr<BPFPreserveAccessIndexAttr>();
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
  } else {
    return arrayAlign.alignmentOfArrayElement(eltSize);
  }
}

static mlir::Value
buildArraySubscriptPtr(CIRGenFunction &CGF, mlir::Location beginLoc,
                       mlir::Location endLoc, mlir::Value ptr, mlir::Type eltTy,
                       ArrayRef<mlir::Value> indices, bool inbounds,
                       bool signedIndices, bool shouldDecay,
                       const llvm::Twine &name = "arrayidx") {
  assert(indices.size() == 1 && "cannot handle multiple indices yet");
  auto idx = indices.back();
  auto &CGM = CGF.getCIRGenModule();
  // TODO(cir): LLVM codegen emits in bound gep check here, is there anything
  // that would enhance tracking this later in CIR?
  if (inbounds)
    assert(!MissingFeatures::emitCheckedInBoundsGEP() && "NYI");
  return CGM.getBuilder().getArrayElement(beginLoc, endLoc, ptr, eltTy, idx,
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
    CIRGenFunction &CGF, mlir::Location beginLoc, mlir::Location endLoc,
    Address addr, ArrayRef<mlir::Value> indices, QualType eltType,
    bool inbounds, bool signedIndices, mlir::Location loc, bool shouldDecay,
    QualType *arrayType = nullptr, const Expr *Base = nullptr,
    const llvm::Twine &name = "arrayidx") {
  // Determine the element size of the statically-sized base.  This is
  // the thing that the indices are expressed in terms of.
  if (auto vla = CGF.getContext().getAsVariableArrayType(eltType)) {
    eltType = getFixedSizeElementType(CGF.getContext(), vla);
  }

  // We can use that to compute the best alignment of the element.
  CharUnits eltSize = CGF.getContext().getTypeSizeInChars(eltType);
  CharUnits eltAlign =
      getArrayElementAlign(addr.getAlignment(), indices.back(), eltSize);

  mlir::Value eltPtr;
  auto LastIndex = getConstantIndexOrNull(indices.back());
  if (!LastIndex ||
      (!CGF.IsInPreservedAIRegion && !isPreserveAIArrayBase(CGF, Base))) {
    eltPtr = buildArraySubscriptPtr(CGF, beginLoc, endLoc, addr.getPointer(),
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

  return Address(eltPtr, CGF.getTypes().convertTypeForMem(eltType), eltAlign);
}

LValue CIRGenFunction::buildArraySubscriptExpr(const ArraySubscriptExpr *E,
                                               bool Accessed) {
  // The index must always be an integer, which is not an aggregate.  Emit it
  // in lexical order (this complexity is, sadly, required by C++17).
  mlir::Value IdxPre =
      (E->getLHS() == E->getIdx()) ? buildScalarExpr(E->getIdx()) : nullptr;
  bool SignedIndices = false;
  auto EmitIdxAfterBase = [&, IdxPre](bool Promote) -> mlir::Value {
    mlir::Value Idx = IdxPre;
    if (E->getLHS() != E->getIdx()) {
      assert(E->getRHS() == E->getIdx() && "index was neither LHS nor RHS");
      Idx = buildScalarExpr(E->getIdx());
    }

    QualType IdxTy = E->getIdx()->getType();
    bool IdxSigned = IdxTy->isSignedIntegerOrEnumerationType();
    SignedIndices |= IdxSigned;

    if (SanOpts.has(SanitizerKind::ArrayBounds))
      llvm_unreachable("array bounds sanitizer is NYI");

    // Extend or truncate the index type to 32 or 64-bits.
    auto ptrTy = mlir::dyn_cast<mlir::cir::PointerType>(Idx.getType());
    if (Promote && ptrTy && mlir::isa<mlir::cir::IntType>(ptrTy.getPointee()))
      llvm_unreachable("index type cast is NYI");

    return Idx;
  };
  IdxPre = nullptr;

  // If the base is a vector type, then we are forming a vector element
  // with this subscript.
  if (E->getBase()->getType()->isVectorType() &&
      !isa<ExtVectorElementExpr>(E->getBase())) {
    LValue lhs = buildLValue(E->getBase());
    auto index = EmitIdxAfterBase(/*Promote=*/false);
    return LValue::MakeVectorElt(lhs.getAddress(), index,
                                 E->getBase()->getType(), lhs.getBaseInfo(),
                                 lhs.getTBAAInfo());
  }

  // All the other cases basically behave like simple offsetting.

  // Handle the extvector case we ignored above.
  if (isa<ExtVectorElementExpr>(E->getBase())) {
    llvm_unreachable("extvector subscript is NYI");
  }

  assert(!MissingFeatures::tbaa() && "TBAA is NYI");
  LValueBaseInfo EltBaseInfo;
  Address Addr = Address::invalid();
  if (const VariableArrayType *vla =
          getContext().getAsVariableArrayType(E->getType())) {
    // The base must be a pointer, which is not an aggregate.  Emit
    // it.  It needs to be emitted first in case it's what captures
    // the VLA bounds.
    Addr = buildPointerWithAlignment(E->getBase(), &EltBaseInfo);
    auto Idx = EmitIdxAfterBase(/*Promote*/ true);

    // The element count here is the total number of non-VLA elements.
    mlir::Value numElements = getVLASize(vla).NumElts;
    Idx = builder.createCast(mlir::cir::CastKind::integral, Idx,
                             numElements.getType());
    Idx = builder.createMul(Idx, numElements);

    QualType ptrType = E->getBase()->getType();
    Addr = buildArraySubscriptPtr(
        *this, CGM.getLoc(E->getBeginLoc()), CGM.getLoc(E->getEndLoc()), Addr,
        {Idx}, E->getType(), !getLangOpts().isSignedOverflowDefined(),
        SignedIndices, CGM.getLoc(E->getExprLoc()), /*shouldDecay=*/false,
        &ptrType, E->getBase());
  } else if (const ObjCObjectType *OIT =
                 E->getType()->getAs<ObjCObjectType>()) {
    llvm_unreachable("ObjC object type subscript is NYI");
  } else if (const Expr *Array = isSimpleArrayDecayOperand(E->getBase())) {
    // If this is A[i] where A is an array, the frontend will have decayed
    // the base to be a ArrayToPointerDecay implicit cast.  While correct, it is
    // inefficient at -O0 to emit a "gep A, 0, 0" when codegen'ing it, then
    // a "gep x, i" here.  Emit one "gep A, 0, i".
    assert(Array->getType()->isArrayType() &&
           "Array to pointer decay must have array source type!");
    LValue ArrayLV;
    // For simple multidimensional array indexing, set the 'accessed' flag
    // for better bounds-checking of the base expression.
    if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(Array))
      ArrayLV = buildArraySubscriptExpr(ASE, /*Accessed=*/true);
    else
      ArrayLV = buildLValue(Array);
    auto Idx = EmitIdxAfterBase(/*Promote=*/true);

    // Propagate the alignment from the array itself to the result.
    QualType arrayType = Array->getType();
    Addr = buildArraySubscriptPtr(
        *this, CGM.getLoc(Array->getBeginLoc()), CGM.getLoc(Array->getEndLoc()),
        ArrayLV.getAddress(), {Idx}, E->getType(),
        !getLangOpts().isSignedOverflowDefined(), SignedIndices,
        CGM.getLoc(E->getExprLoc()), /*shouldDecay=*/true, &arrayType,
        E->getBase());
    EltBaseInfo = ArrayLV.getBaseInfo();
    // TODO(cir): EltTBAAInfo
    assert(!MissingFeatures::tbaa() && "TBAA is NYI");
  } else {
    // The base must be a pointer; emit it with an estimate of its alignment.
    // TODO(cir): EltTBAAInfo
    assert(!MissingFeatures::tbaa() && "TBAA is NYI");
    Addr = buildPointerWithAlignment(E->getBase(), &EltBaseInfo);
    auto Idx = EmitIdxAfterBase(/*Promote*/ true);
    QualType ptrType = E->getBase()->getType();
    Addr = buildArraySubscriptPtr(
        *this, CGM.getLoc(E->getBeginLoc()), CGM.getLoc(E->getEndLoc()), Addr,
        Idx, E->getType(), !getLangOpts().isSignedOverflowDefined(),
        SignedIndices, CGM.getLoc(E->getExprLoc()), /*shouldDecay=*/false,
        &ptrType, E->getBase());
  }

  LValue LV = LValue::makeAddr(Addr, E->getType(), EltBaseInfo);

  if (getLangOpts().ObjC && getLangOpts().getGC() != LangOptions::NonGC) {
    llvm_unreachable("ObjC is NYI");
  }

  return LV;
}

LValue CIRGenFunction::buildStringLiteralLValue(const StringLiteral *E) {
  auto sym = CGM.getAddrOfConstantStringFromLiteral(E).getSymbol();

  auto cstGlobal = mlir::SymbolTable::lookupSymbolIn(CGM.getModule(), sym);
  assert(cstGlobal && "Expected global");

  auto g = dyn_cast<mlir::cir::GlobalOp>(cstGlobal);
  assert(g && "unaware of other symbol providers");

  auto ptrTy = mlir::cir::PointerType::get(CGM.getBuilder().getContext(),
                                           g.getSymType());
  assert(g.getAlignment() && "expected alignment for string literal");
  auto align = *g.getAlignment();
  auto addr = builder.create<mlir::cir::GetGlobalOp>(
      getLoc(E->getSourceRange()), ptrTy, g.getSymName());
  return makeAddrLValue(
      Address(addr, g.getSymType(), CharUnits::fromQuantity(align)),
      E->getType(), AlignmentSource::Decl);
}

/// Casts are never lvalues unless that cast is to a reference type. If the cast
/// is to a reference, we can have the usual lvalue result, otherwise if a cast
/// is needed by the code generator in an lvalue context, then it must mean that
/// we need the address of an aggregate in order to access one of its members.
/// This can happen for all the reasons that casts are permitted with aggregate
/// result, including noop aggregate casts, and cast from scalar to union.
LValue CIRGenFunction::buildCastLValue(const CastExpr *E) {
  switch (E->getCastKind()) {
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
    LValue LV = buildLValue(E->getSubExpr());
    Address V = LV.getAddress();
    const auto *DCE = cast<CXXDynamicCastExpr>(E);
    return MakeNaturalAlignAddrLValue(buildDynamicCast(V, DCE), E->getType());
  }

  case CK_ConstructorConversion:
  case CK_UserDefinedConversion:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_LValueToRValue:
    return buildLValue(E->getSubExpr());

  case CK_NoOp: {
    // CK_NoOp can model a qualification conversion, which can remove an array
    // bound and change the IR type.
    LValue LV = buildLValue(E->getSubExpr());
    if (LV.isSimple()) {
      Address V = LV.getAddress();
      if (V.isValid()) {
        auto T = getTypes().convertTypeForMem(E->getType());
        if (V.getElementType() != T)
          LV.setAddress(
              builder.createElementBitCast(getLoc(E->getSourceRange()), V, T));
      }
    }
    return LV;
  }

  case CK_UncheckedDerivedToBase:
  case CK_DerivedToBase: {
    const auto *DerivedClassTy =
        E->getSubExpr()->getType()->castAs<RecordType>();
    auto *DerivedClassDecl = cast<CXXRecordDecl>(DerivedClassTy->getDecl());

    LValue LV = buildLValue(E->getSubExpr());
    Address This = LV.getAddress();

    // Perform the derived-to-base conversion
    Address Base = getAddressOfBaseClass(
        This, DerivedClassDecl, E->path_begin(), E->path_end(),
        /*NullCheckValue=*/false, E->getExprLoc());

    // TODO: Support accesses to members of base classes in TBAA. For now, we
    // conservatively pretend that the complete object is of the base class
    // type.
    assert(!MissingFeatures::tbaa());
    return makeAddrLValue(Base, E->getType(), LV.getBaseInfo());
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
    LValue LV = buildLValue(E->getSubExpr());
    QualType DestTy = getContext().getPointerType(E->getType());
    auto SrcAS =
        builder.getAddrSpaceAttr(E->getSubExpr()->getType().getAddressSpace());
    auto DestAS = builder.getAddrSpaceAttr(E->getType().getAddressSpace());
    mlir::Value V = getTargetHooks().performAddrSpaceCast(
        *this, LV.getPointer(), SrcAS, DestAS, ConvertType(DestTy));
    assert(!MissingFeatures::tbaa());
    return makeAddrLValue(Address(V, getTypes().convertTypeForMem(E->getType()),
                                  LV.getAddress().getAlignment()),
                          E->getType(), LV.getBaseInfo());
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
static DeclRefExpr *tryToConvertMemberExprToDeclRefExpr(CIRGenFunction &CGF,
                                                        const MemberExpr *ME) {
  if (auto *VD = dyn_cast<VarDecl>(ME->getMemberDecl())) {
    // Try to emit static variable member expressions as DREs.
    return DeclRefExpr::Create(
        CGF.getContext(), NestedNameSpecifierLoc(), SourceLocation(), VD,
        /*RefersToEnclosingVariableOrCapture=*/false, ME->getExprLoc(),
        ME->getType(), ME->getValueKind(), nullptr, nullptr, ME->isNonOdrUse());
  }
  return nullptr;
}

LValue CIRGenFunction::buildCheckedLValue(const Expr *E, TypeCheckKind TCK) {
  LValue LV;
  if (SanOpts.has(SanitizerKind::ArrayBounds) && isa<ArraySubscriptExpr>(E))
    assert(0 && "not implemented");
  else
    LV = buildLValue(E);
  if (!isa<DeclRefExpr>(E) && !LV.isBitField() && LV.isSimple()) {
    SanitizerSet SkippedChecks;
    if (const auto *ME = dyn_cast<MemberExpr>(E)) {
      bool IsBaseCXXThis = isWrappedCXXThis(ME->getBase());
      if (IsBaseCXXThis)
        SkippedChecks.set(SanitizerKind::Alignment, true);
      if (IsBaseCXXThis || isa<DeclRefExpr>(ME->getBase()))
        SkippedChecks.set(SanitizerKind::Null, true);
    }
    buildTypeCheck(TCK, E->getExprLoc(), LV.getPointer(), E->getType(),
                   LV.getAlignment(), SkippedChecks);
  }
  return LV;
}

// TODO(cir): candidate for common AST helper for LLVM and CIR codegen
bool CIRGenFunction::isWrappedCXXThis(const Expr *Obj) {
  const Expr *Base = Obj;
  while (!isa<CXXThisExpr>(Base)) {
    // The result of a dynamic_cast can be null.
    if (isa<CXXDynamicCastExpr>(Base))
      return false;

    if (const auto *CE = dyn_cast<CastExpr>(Base)) {
      Base = CE->getSubExpr();
    } else if (const auto *PE = dyn_cast<ParenExpr>(Base)) {
      Base = PE->getSubExpr();
    } else if (const auto *UO = dyn_cast<UnaryOperator>(Base)) {
      if (UO->getOpcode() == UO_Extension)
        Base = UO->getSubExpr();
      else
        return false;
    } else {
      return false;
    }
  }
  return true;
}

LValue CIRGenFunction::buildMemberExpr(const MemberExpr *E) {
  if (DeclRefExpr *DRE = tryToConvertMemberExprToDeclRefExpr(*this, E)) {
    buildIgnoredExpr(E->getBase());
    return buildDeclRefLValue(DRE);
  }

  Expr *BaseExpr = E->getBase();
  // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a scalar.
  LValue BaseLV;
  if (E->isArrow()) {
    LValueBaseInfo BaseInfo;
    Address Addr = buildPointerWithAlignment(BaseExpr, &BaseInfo);
    QualType PtrTy = BaseExpr->getType()->getPointeeType();
    SanitizerSet SkippedChecks;
    bool IsBaseCXXThis = isWrappedCXXThis(BaseExpr);
    if (IsBaseCXXThis)
      SkippedChecks.set(SanitizerKind::Alignment, true);
    if (IsBaseCXXThis || isa<DeclRefExpr>(BaseExpr))
      SkippedChecks.set(SanitizerKind::Null, true);
    buildTypeCheck(TCK_MemberAccess, E->getExprLoc(), Addr.getPointer(), PtrTy,
                   /*Alignment=*/CharUnits::Zero(), SkippedChecks);
    BaseLV = makeAddrLValue(Addr, PtrTy, BaseInfo);
  } else
    BaseLV = buildCheckedLValue(BaseExpr, TCK_MemberAccess);

  NamedDecl *ND = E->getMemberDecl();
  if (auto *Field = dyn_cast<FieldDecl>(ND)) {
    LValue LV = buildLValueForField(BaseLV, Field);
    assert(!MissingFeatures::setObjCGCLValueClass() && "NYI");
    if (getLangOpts().OpenMP) {
      // If the member was explicitly marked as nontemporal, mark it as
      // nontemporal. If the base lvalue is marked as nontemporal, mark access
      // to children as nontemporal too.
      assert(0 && "not implemented");
    }
    return LV;
  }

  if (const auto *FD = dyn_cast<FunctionDecl>(ND))
    assert(0 && "not implemented");

  llvm_unreachable("Unhandled member declaration!");
}

LValue CIRGenFunction::buildCallExprLValue(const CallExpr *E) {
  RValue RV = buildCallExpr(E);

  if (!RV.isScalar())
    return makeAddrLValue(RV.getAggregateAddress(), E->getType(),
                          AlignmentSource::Decl);

  assert(E->getCallReturnType(getContext())->isReferenceType() &&
         "Can't have a scalar return unless the return type is a "
         "reference type!");

  return MakeNaturalAlignPointeeAddrLValue(RV.getScalarVal(), E->getType());
}

/// Evaluate an expression into a given memory location.
void CIRGenFunction::buildAnyExprToMem(const Expr *E, Address Location,
                                       Qualifiers Quals, bool IsInit) {
  // FIXME: This function should take an LValue as an argument.
  switch (getEvaluationKind(E->getType())) {
  case TEK_Complex:
    assert(0 && "NYI");
    return;

  case TEK_Aggregate: {
    buildAggExpr(E, AggValueSlot::forAddr(Location, Quals,
                                          AggValueSlot::IsDestructed_t(IsInit),
                                          AggValueSlot::DoesNotNeedGCBarriers,
                                          AggValueSlot::IsAliased_t(!IsInit),
                                          AggValueSlot::MayOverlap));
    return;
  }

  case TEK_Scalar: {
    RValue RV = RValue::get(buildScalarExpr(E));
    LValue LV = makeAddrLValue(Location, E->getType());
    buildStoreThroughLValue(RV, LV);
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}

static Address createReferenceTemporary(CIRGenFunction &CGF,
                                        const MaterializeTemporaryExpr *M,
                                        const Expr *Inner,
                                        Address *Alloca = nullptr) {
  // TODO(cir): CGF.getTargetHooks();
  switch (M->getStorageDuration()) {
  case SD_FullExpression:
  case SD_Automatic: {
    // TODO(cir): probably not needed / too LLVM specific?
    // If we have a constant temporary array or record try to promote it into a
    // constant global under the same rules a normal constant would've been
    // promoted. This is easier on the optimizer and generally emits fewer
    // instructions.
    QualType Ty = Inner->getType();
    if (CGF.CGM.getCodeGenOpts().MergeAllConstants &&
        (Ty->isArrayType() || Ty->isRecordType()) &&
        CGF.CGM.isTypeConstant(Ty, /*ExcludeCtor=*/true, /*ExcludeDtor=*/false))
      assert(0 && "NYI");

    // The temporary memory should be created in the same scope as the extending
    // declaration of the temporary materialization expression.
    mlir::cir::AllocaOp extDeclAlloca;
    if (const clang::ValueDecl *extDecl = M->getExtendingDecl()) {
      auto extDeclAddrIter = CGF.LocalDeclMap.find(extDecl);
      if (extDeclAddrIter != CGF.LocalDeclMap.end()) {
        extDeclAlloca = dyn_cast_if_present<mlir::cir::AllocaOp>(
            extDeclAddrIter->second.getDefiningOp());
      }
    }
    mlir::OpBuilder::InsertPoint ip;
    if (extDeclAlloca)
      ip = {extDeclAlloca->getBlock(), extDeclAlloca->getIterator()};
    return CGF.CreateMemTemp(Ty, CGF.getLoc(M->getSourceRange()),
                             CGF.getCounterRefTmpAsString(), Alloca, ip);
  }
  case SD_Thread:
  case SD_Static:
    assert(0 && "NYI");

  case SD_Dynamic:
    llvm_unreachable("temporary can't have dynamic storage duration");
  }
  llvm_unreachable("unknown storage duration");
}

static void pushTemporaryCleanup(CIRGenFunction &CGF,
                                 const MaterializeTemporaryExpr *M,
                                 const Expr *E, Address ReferenceTemporary) {
  // Objective-C++ ARC:
  //   If we are binding a reference to a temporary that has ownership, we
  //   need to perform retain/release operations on the temporary.
  //
  // FIXME: This should be looking at E, not M.
  if (auto Lifetime = M->getType().getObjCLifetime()) {
    assert(0 && "NYI");
  }

  CXXDestructorDecl *ReferenceTemporaryDtor = nullptr;
  if (const RecordType *RT =
          E->getType()->getBaseElementTypeUnsafe()->getAs<RecordType>()) {
    // Get the destructor for the reference temporary.
    auto *ClassDecl = cast<CXXRecordDecl>(RT->getDecl());
    if (!ClassDecl->hasTrivialDestructor())
      ReferenceTemporaryDtor = ClassDecl->getDestructor();
  }

  if (!ReferenceTemporaryDtor)
    return;

  // Call the destructor for the temporary.
  switch (M->getStorageDuration()) {
  case SD_Static:
  case SD_Thread: {
    if (E->getType()->isArrayType()) {
      llvm_unreachable("SD_Static|SD_Thread + array types not implemented");
    } else {
      llvm_unreachable("SD_Static|SD_Thread for general types not implemented");
    }
    llvm_unreachable("SD_Static|SD_Thread not implemented");
  }

  case SD_FullExpression:
    CGF.pushDestroy(NormalAndEHCleanup, ReferenceTemporary, E->getType(),
                    CIRGenFunction::destroyCXXObject,
                    CGF.getLangOpts().Exceptions);
    break;

  case SD_Automatic:
    llvm_unreachable("SD_Automatic not implemented");
    break;

  case SD_Dynamic:
    llvm_unreachable("temporary cannot have dynamic storage duration");
  }
}

LValue CIRGenFunction::buildMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *M) {
  const Expr *E = M->getSubExpr();

  assert((!M->getExtendingDecl() || !isa<VarDecl>(M->getExtendingDecl()) ||
          !cast<VarDecl>(M->getExtendingDecl())->isARCPseudoStrong()) &&
         "Reference should never be pseudo-strong!");

  // FIXME: ideally this would use buildAnyExprToMem, however, we cannot do so
  // as that will cause the lifetime adjustment to be lost for ARC
  auto ownership = M->getType().getObjCLifetime();
  if (ownership != Qualifiers::OCL_None &&
      ownership != Qualifiers::OCL_ExplicitNone) {
    assert(0 && "NYI");
  }

  SmallVector<const Expr *, 2> CommaLHSs;
  SmallVector<SubobjectAdjustment, 2> Adjustments;
  E = E->skipRValueSubobjectAdjustments(CommaLHSs, Adjustments);

  for (const auto &Ignored : CommaLHSs)
    buildIgnoredExpr(Ignored);

  if (const auto *opaque = dyn_cast<OpaqueValueExpr>(E))
    assert(0 && "NYI");

  // Create and initialize the reference temporary.
  Address Alloca = Address::invalid();
  Address Object = createReferenceTemporary(*this, M, E, &Alloca);

  if (auto Var =
          dyn_cast<mlir::cir::GlobalOp>(Object.getPointer().getDefiningOp())) {
    // TODO(cir): add something akin to stripPointerCasts() to ptr above
    assert(0 && "NYI");
  } else {
    switch (M->getStorageDuration()) {
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

    buildAnyExprToMem(E, Object, Qualifiers(), /*IsInit*/ true);
  }
  pushTemporaryCleanup(*this, M, E, Object);

  // Perform derived-to-base casts and/or field accesses, to get from the
  // temporary object we created (and, potentially, for which we extended
  // the lifetime) to the subobject we're binding the reference to.
  for (SubobjectAdjustment &Adjustment : llvm::reverse(Adjustments)) {
    (void)Adjustment;
    assert(0 && "NYI");
  }

  return makeAddrLValue(Object, M->getType(), AlignmentSource::Decl);
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
std::optional<LValue> HandleConditionalOperatorLValueSimpleCase(
    CIRGenFunction &CGF, const AbstractConditionalOperator *E) {
  const Expr *condExpr = E->getCond();
  bool CondExprBool;
  if (CGF.ConstantFoldsToSimpleInteger(condExpr, CondExprBool)) {
    const Expr *Live = E->getTrueExpr(), *Dead = E->getFalseExpr();
    if (!CondExprBool)
      std::swap(Live, Dead);

    if (!CGF.ContainsLabel(Dead)) {
      // If the true case is live, we need to track its region.
      if (CondExprBool) {
        assert(!MissingFeatures::incrementProfileCounter());
      }
      // If a throw expression we emit it and return an undefined lvalue
      // because it can't be used.
      if (auto *ThrowExpr = dyn_cast<CXXThrowExpr>(Live->IgnoreParens())) {
        llvm_unreachable("NYI");
      }
      return CGF.buildLValue(Live);
    }
  }
  return std::nullopt;
}
} // namespace

/// Emit the operand of a glvalue conditional operator. This is either a glvalue
/// or a (possibly-parenthesized) throw-expression. If this is a throw, no
/// LValue is returned and the current block has been terminated.
static std::optional<LValue> buildLValueOrThrowExpression(CIRGenFunction &CGF,
                                                          const Expr *Operand) {
  if (auto *ThrowExpr = dyn_cast<CXXThrowExpr>(Operand->IgnoreParens())) {
    llvm_unreachable("NYI");
  }

  return CGF.buildLValue(Operand);
}

// Create and generate the 3 blocks for a conditional operator.
// Leaves the 'current block' in the continuation basic block.
template <typename FuncTy>
CIRGenFunction::ConditionalInfo
CIRGenFunction::buildConditionalBlocks(const AbstractConditionalOperator *E,
                                       const FuncTy &BranchGenFunc) {
  ConditionalInfo Info;
  auto &CGF = *this;
  ConditionalEvaluation eval(CGF);
  auto loc = CGF.getLoc(E->getSourceRange());
  auto &builder = CGF.getBuilder();
  auto *trueExpr = E->getTrueExpr();
  auto *falseExpr = E->getFalseExpr();

  mlir::Value condV = CGF.buildOpOnBoolExpr(loc, E->getCond());
  SmallVector<mlir::OpBuilder::InsertPoint, 2> insertPoints{};
  mlir::Type yieldTy{};

  auto patchVoidOrThrowSites = [&]() {
    if (insertPoints.empty())
      return;
    // If both arms are void, so be it.
    if (!yieldTy)
      yieldTy = CGF.VoidTy;

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

  Info.Result = builder
                    .create<mlir::cir::TernaryOp>(
                        loc, condV, /*trueBuilder=*/
                        [&](mlir::OpBuilder &b, mlir::Location loc) {
                          CIRGenFunction::LexicalScope lexScope{
                              *this, loc, b.getInsertionBlock()};
                          CGF.currLexScope->setAsTernary();

                          assert(!MissingFeatures::incrementProfileCounter());
                          eval.begin(CGF);
                          Info.LHS = BranchGenFunc(CGF, trueExpr);
                          auto lhs = Info.LHS->getPointer();
                          eval.end(CGF);

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
                          CGF.currLexScope->setAsTernary();

                          assert(!MissingFeatures::incrementProfileCounter());
                          eval.begin(CGF);
                          Info.RHS = BranchGenFunc(CGF, falseExpr);
                          auto rhs = Info.RHS->getPointer();
                          eval.end(CGF);

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
  return Info;
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
  if (std::optional<LValue> Res =
          HandleConditionalOperatorLValueSimpleCase(*this, expr))
    return *Res;

  ConditionalInfo Info =
      buildConditionalBlocks(expr, [](CIRGenFunction &CGF, const Expr *E) {
        return buildLValueOrThrowExpression(CGF, E);
      });

  if ((Info.LHS && !Info.LHS->isSimple()) ||
      (Info.RHS && !Info.RHS->isSimple()))
    llvm_unreachable("unsupported conditional operator");

  if (Info.LHS && Info.RHS) {
    Address lhsAddr = Info.LHS->getAddress();
    Address rhsAddr = Info.RHS->getAddress();
    Address result(Info.Result, lhsAddr.getElementType(),
                   std::min(lhsAddr.getAlignment(), rhsAddr.getAlignment()));
    AlignmentSource alignSource =
        std::max(Info.LHS->getBaseInfo().getAlignmentSource(),
                 Info.RHS->getBaseInfo().getAlignmentSource());
    assert(!MissingFeatures::tbaa());
    return makeAddrLValue(result, expr->getType(), LValueBaseInfo(alignSource));
  } else {
    llvm_unreachable("NYI");
  }
}

/// Emit code to compute a designator that specifies the location
/// of the expression.
/// FIXME: document this function better.
LValue CIRGenFunction::buildLValue(const Expr *E) {
  // FIXME: ApplyDebugLocation DL(*this, E);
  switch (E->getStmtClass()) {
  default: {
    emitError(getLoc(E->getExprLoc()), "l-value not implemented for '")
        << E->getStmtClassName() << "'";
    assert(0 && "not implemented");
  }
  case Expr::ConditionalOperatorClass:
    return buildConditionalOperatorLValue(cast<ConditionalOperator>(E));
  case Expr::ArraySubscriptExprClass:
    return buildArraySubscriptExpr(cast<ArraySubscriptExpr>(E));
  case Expr::ExtVectorElementExprClass:
    return buildExtVectorElementExpr(cast<ExtVectorElementExpr>(E));
  case Expr::BinaryOperatorClass:
    return buildBinaryOperatorLValue(cast<BinaryOperator>(E));
  case Expr::CompoundAssignOperatorClass: {
    QualType Ty = E->getType();
    if (const AtomicType *AT = Ty->getAs<AtomicType>())
      assert(0 && "not yet implemented");
    if (!Ty->isAnyComplexType())
      return buildCompoundAssignmentLValue(cast<CompoundAssignOperator>(E));
    return buildComplexCompoundAssignmentLValue(
        cast<CompoundAssignOperator>(E));
  }
  case Expr::CallExprClass:
  case Expr::CXXMemberCallExprClass:
  case Expr::CXXOperatorCallExprClass:
  case Expr::UserDefinedLiteralClass:
    return buildCallExprLValue(cast<CallExpr>(E));
  case Expr::ExprWithCleanupsClass: {
    const auto *cleanups = cast<ExprWithCleanups>(E);
    LValue LV;

    auto scopeLoc = getLoc(E->getSourceRange());
    [[maybe_unused]] auto scope = builder.create<mlir::cir::ScopeOp>(
        scopeLoc, /*scopeBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          CIRGenFunction::LexicalScope lexScope{*this, loc,
                                                builder.getInsertionBlock()};

          LV = buildLValue(cleanups->getSubExpr());
          if (LV.isSimple()) {
            // Defend against branches out of gnu statement expressions
            // surrounded by cleanups.
            Address addr = LV.getAddress();
            auto v = addr.getPointer();
            LV = LValue::makeAddr(addr.withPointer(v, NotKnownNonNull),
                                  LV.getType(), getContext(), LV.getBaseInfo(),
                                  LV.getTBAAInfo());
          }
        });

    // FIXME: Is it possible to create an ExprWithCleanups that produces a
    // bitfield lvalue or some other non-simple lvalue?
    return LV;
  }
  case Expr::ParenExprClass:
    return buildLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::DeclRefExprClass:
    return buildDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::UnaryOperatorClass:
    return buildUnaryOpLValue(cast<UnaryOperator>(E));
  case Expr::StringLiteralClass:
    return buildStringLiteralLValue(cast<StringLiteral>(E));
  case Expr::MemberExprClass:
    return buildMemberExpr(cast<MemberExpr>(E));
  case Expr::CompoundLiteralExprClass:
    return buildCompoundLiteralLValue(cast<CompoundLiteralExpr>(E));
  case Expr::PredefinedExprClass:
    return buildPredefinedLValue(cast<PredefinedExpr>(E));
  case Expr::CXXFunctionalCastExprClass:
  case Expr::CXXReinterpretCastExprClass:
  case Expr::CXXConstCastExprClass:
  case Expr::CXXAddrspaceCastExprClass:
  case Expr::ObjCBridgedCastExprClass:
    emitError(getLoc(E->getExprLoc()), "l-value not implemented for '")
        << E->getStmtClassName() << "'";
    assert(0 && "Use buildCastLValue below, remove me when adding testcase");
  case Expr::CStyleCastExprClass:
  case Expr::CXXStaticCastExprClass:
  case Expr::CXXDynamicCastExprClass:
  case Expr::ImplicitCastExprClass:
    return buildCastLValue(cast<CastExpr>(E));
  case Expr::OpaqueValueExprClass:
    return buildOpaqueValueLValue(cast<OpaqueValueExpr>(E));

  case Expr::MaterializeTemporaryExprClass:
    return buildMaterializeTemporaryExpr(cast<MaterializeTemporaryExpr>(E));

  case Expr::ObjCPropertyRefExprClass:
    llvm_unreachable("cannot emit a property reference directly");
  case Expr::StmtExprClass:
    return buildStmtExprLValue(cast<StmtExpr>(E));
  }

  return LValue::makeAddr(Address::invalid(), E->getType());
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
bool CIRGenFunction::LValueIsSuitableForInlineAtomic(LValue LV) {
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

  if (const UnaryOperator *CondUOp = dyn_cast<UnaryOperator>(cond)) {
    // In LLVM the condition is reversed here for efficient codegen.
    // This should be done in CIR prior to LLVM lowering, if we do now
    // we can make CIR based diagnostics misleading.
    //  cir.ternary(!x, t, f) -> cir.ternary(x, f, t)
    assert(!MissingFeatures::shouldReverseUnaryCondOnBoolExpr());
  }

  if (const ConditionalOperator *CondOp = dyn_cast<ConditionalOperator>(cond)) {
    auto *trueExpr = CondOp->getTrueExpr();
    auto *falseExpr = CondOp->getFalseExpr();
    mlir::Value condV = buildOpOnBoolExpr(loc, CondOp->getCond());

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

    return buildScalarConversion(ternaryOpRes, CondOp->getType(),
                                 getContext().BoolTy, CondOp->getExprLoc());
  }

  if (const CXXThrowExpr *Throw = dyn_cast<CXXThrowExpr>(cond)) {
    llvm_unreachable("NYI");
  }

  // If the branch has a condition wrapped by __builtin_unpredictable,
  // create metadata that specifies that the branch is unpredictable.
  // Don't bother if not optimizing because that metadata would not be used.
  auto *Call = dyn_cast<CallExpr>(cond->IgnoreImpCasts());
  if (Call && CGM.getCodeGenOpts().OptimizationLevel != 0) {
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

mlir::Value CIRGenFunction::buildFromMemory(mlir::Value Value, QualType Ty) {
  if (!Ty->isBooleanType() && hasBooleanRepresentation(Ty)) {
    llvm_unreachable("NIY");
  }

  return Value;
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

  auto ElemTy = addr.getElementType();

  if (const auto *ClangVecTy = ty->getAs<clang::VectorType>()) {
    // Handle vectors of size 3 like size 4 for better performance.
    const auto VTy = cast<mlir::cir::VectorType>(ElemTy);

    if (!CGM.getCodeGenOpts().PreserveVec3Type &&
        ClangVecTy->getNumElements() == 3) {
      auto loc = addr.getPointer().getLoc();
      auto vec4Ty =
          mlir::cir::VectorType::get(VTy.getContext(), VTy.getEltType(), 4);
      Address Cast = addr.withElementType(vec4Ty);
      // Now load value.
      mlir::Value V = builder.createLoad(loc, Cast);

      // Shuffle vector to get vec3.
      V = builder.createVecShuffle(loc, V, ArrayRef<int64_t>{0, 1, 2});
      return buildFromMemory(V, ty);
    }
  }

  auto Ptr = addr.getPointer();
  if (mlir::isa<mlir::cir::VoidType>(ElemTy)) {
    ElemTy = mlir::cir::IntType::get(builder.getContext(), 8, true);
    auto ElemPtrTy = mlir::cir::PointerType::get(builder.getContext(), ElemTy);
    Ptr = builder.create<mlir::cir::CastOp>(loc, ElemPtrTy,
                                            mlir::cir::CastKind::bitcast, Ptr);
  }

  mlir::Value Load = builder.CIRBaseBuilderTy::createLoad(loc, Ptr, isVolatile);

  if (isNontemporal) {
    llvm_unreachable("NYI");
  }

  assert(!MissingFeatures::tbaa() && "NYI");
  assert(!MissingFeatures::emitScalarRangeCheck() && "NYI");

  return buildFromMemory(Load, ty);
}

// Note: this function also emit constructor calls to support a MSVC extensions
// allowing explicit constructor function call.
RValue CIRGenFunction::buildCXXMemberCallExpr(const CXXMemberCallExpr *CE,
                                              ReturnValueSlot ReturnValue) {

  const Expr *callee = CE->getCallee()->IgnoreParens();

  if (isa<BinaryOperator>(callee))
    return buildCXXMemberPointerCallExpr(CE, ReturnValue);

  const auto *ME = cast<MemberExpr>(callee);
  const auto *MD = cast<CXXMethodDecl>(ME->getMemberDecl());

  if (MD->isStatic()) {
    llvm_unreachable("NYI");
  }

  bool HasQualifier = ME->hasQualifier();
  NestedNameSpecifier *Qualifier = HasQualifier ? ME->getQualifier() : nullptr;
  bool IsArrow = ME->isArrow();
  const Expr *Base = ME->getBase();

  return buildCXXMemberOrOperatorMemberCallExpr(
      CE, MD, ReturnValue, HasQualifier, Qualifier, IsArrow, Base);
}

RValue CIRGenFunction::buildReferenceBindingToExpr(const Expr *E) {
  // Emit the expression as an lvalue.
  LValue LV = buildLValue(E);
  assert(LV.isSimple());
  auto Value = LV.getPointer();

  if (sanitizePerformTypeCheck() && !E->getType()->isFunctionType()) {
    assert(0 && "NYI");
  }

  return RValue::get(Value);
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

LValue CIRGenFunction::buildLoadOfReferenceLValue(LValue RefLVal,
                                                  mlir::Location Loc) {
  LValueBaseInfo PointeeBaseInfo;
  Address PointeeAddr = buildLoadOfReference(RefLVal, Loc, &PointeeBaseInfo);
  return makeAddrLValue(PointeeAddr, RefLVal.getType()->getPointeeType(),
                        PointeeBaseInfo);
}

void CIRGenFunction::buildUnreachable(SourceLocation Loc) {
  if (SanOpts.has(SanitizerKind::Unreachable))
    llvm_unreachable("NYI");
  builder.create<mlir::cir::UnreachableOp>(getLoc(Loc));
}

//===----------------------------------------------------------------------===//
// CIR builder helpers
//===----------------------------------------------------------------------===//

Address CIRGenFunction::CreateMemTemp(QualType Ty, mlir::Location Loc,
                                      const Twine &Name, Address *Alloca,
                                      mlir::OpBuilder::InsertPoint ip) {
  // FIXME: Should we prefer the preferred type alignment here?
  return CreateMemTemp(Ty, getContext().getTypeAlignInChars(Ty), Loc, Name,
                       Alloca, ip);
}

Address CIRGenFunction::CreateMemTemp(QualType Ty, CharUnits Align,
                                      mlir::Location Loc, const Twine &Name,
                                      Address *Alloca,
                                      mlir::OpBuilder::InsertPoint ip) {
  Address Result =
      CreateTempAlloca(getTypes().convertTypeForMem(Ty), Align, Loc, Name,
                       /*ArraySize=*/nullptr, Alloca, ip);
  if (Ty->isConstantMatrixType()) {
    assert(0 && "NYI");
  }
  return Result;
}

/// This creates a alloca and inserts it into the entry block of the
/// current region.
Address CIRGenFunction::CreateTempAllocaWithoutCast(
    mlir::Type Ty, CharUnits Align, mlir::Location Loc, const Twine &Name,
    mlir::Value ArraySize, mlir::OpBuilder::InsertPoint ip) {
  auto Alloca = ip.isSet() ? CreateTempAlloca(Ty, Loc, Name, ip, ArraySize)
                           : CreateTempAlloca(Ty, Loc, Name, ArraySize);
  Alloca.setAlignmentAttr(CGM.getSize(Align));
  return Address(Alloca, Ty, Align);
}

/// This creates a alloca and inserts it into the entry block. The alloca is
/// casted to default address space if necessary.
Address CIRGenFunction::CreateTempAlloca(mlir::Type Ty, CharUnits Align,
                                         mlir::Location Loc, const Twine &Name,
                                         mlir::Value ArraySize,
                                         Address *AllocaAddr,
                                         mlir::OpBuilder::InsertPoint ip) {
  auto Alloca =
      CreateTempAllocaWithoutCast(Ty, Align, Loc, Name, ArraySize, ip);
  if (AllocaAddr)
    *AllocaAddr = Alloca;
  mlir::Value V = Alloca.getPointer();
  // Alloca always returns a pointer in alloca address space, which may
  // be different from the type defined by the language. For example,
  // in C++ the auto variables are in the default address space. Therefore
  // cast alloca to the default address space when necessary.
  if (auto ASTAS =
          builder.getAddrSpaceAttr(CGM.getLangTempAllocaAddressSpace());
      getCIRAllocaAddressSpace() != ASTAS) {
    llvm_unreachable("Requires address space cast which is NYI");
  }
  return Address(V, Ty, Align);
}

/// This creates an alloca and inserts it into the entry block if \p ArraySize
/// is nullptr, otherwise inserts it at the current insertion point of the
/// builder.
mlir::cir::AllocaOp
CIRGenFunction::CreateTempAlloca(mlir::Type Ty, mlir::Location Loc,
                                 const Twine &Name, mlir::Value ArraySize,
                                 bool insertIntoFnEntryBlock) {
  return cast<mlir::cir::AllocaOp>(buildAlloca(Name.str(), Ty, Loc, CharUnits(),
                                               insertIntoFnEntryBlock,
                                               ArraySize)
                                       .getDefiningOp());
}

/// This creates an alloca and inserts it into the provided insertion point
mlir::cir::AllocaOp CIRGenFunction::CreateTempAlloca(
    mlir::Type Ty, mlir::Location Loc, const Twine &Name,
    mlir::OpBuilder::InsertPoint ip, mlir::Value ArraySize) {
  assert(ip.isSet() && "Insertion point is not set");
  return cast<mlir::cir::AllocaOp>(
      buildAlloca(Name.str(), Ty, Loc, CharUnits(), ip, ArraySize)
          .getDefiningOp());
}

/// Just like CreateTempAlloca above, but place the alloca into the function
/// entry basic block instead.
mlir::cir::AllocaOp CIRGenFunction::CreateTempAllocaInFnEntryBlock(
    mlir::Type Ty, mlir::Location Loc, const Twine &Name,
    mlir::Value ArraySize) {
  return CreateTempAlloca(Ty, Loc, Name, ArraySize,
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
  if (const auto *RT = dyn_cast<RecordType>(type))
    if (const auto *RD = dyn_cast<CXXRecordDecl>(RT->getDecl()))
      if (RD->hasMutableFields() || !RD->isTrivial())
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
  ConstantEmissionKind CEK;
  if (isa<ParmVarDecl>(value)) {
    CEK = CEK_None;
  } else if (auto *var = dyn_cast<VarDecl>(value)) {
    CEK = checkVarTypeForConstantEmission(var->getType());
  } else if (isa<EnumConstantDecl>(value)) {
    CEK = CEK_AsValueOnly;
  } else {
    CEK = CEK_None;
  }
  if (CEK == CEK_None)
    return ConstantEmission();

  Expr::EvalResult result;
  bool resultIsReference;
  QualType resultType;

  // It's best to evaluate all the way as an r-value if that's permitted.
  if (CEK != CEK_AsReferenceOnly &&
      refExpr->EvaluateAsRValue(result, getContext())) {
    resultIsReference = false;
    resultType = refExpr->getType();

    // Otherwise, try to evaluate as an l-value.
  } else if (CEK != CEK_AsValueOnly &&
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
    auto *MD = dyn_cast_or_null<CXXMethodDecl>(CurCodeDecl);
    if (MD && MD->getParent()->isLambda() &&
        MD->getOverloadedOperator() == OO_Call) {
      const APValue::LValueBase &base = result.Val.getLValueBase();
      if (const ValueDecl *D = base.dyn_cast<const ValueDecl *>()) {
        if (const VarDecl *VD = dyn_cast<const VarDecl>(D)) {
          if (!VD->hasAttr<CUDADeviceAttr>()) {
            return ConstantEmission();
          }
        }
      }
    }
  }

  // Emit as a constant.
  // FIXME(cir): have emitAbstract build a TypedAttr instead (this requires
  // somewhat heavy refactoring...)
  auto C = ConstantEmitter(*this).emitAbstract(refExpr->getLocation(),
                                               result.Val, resultType);
  mlir::TypedAttr cstToEmit = mlir::dyn_cast_if_present<mlir::TypedAttr>(C);
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
CIRGenFunction::tryEmitAsConstant(const MemberExpr *ME) {
  llvm_unreachable("NYI");
}

mlir::Value CIRGenFunction::buildScalarConstant(
    const CIRGenFunction::ConstantEmission &Constant, Expr *E) {
  assert(Constant && "not a constant");
  if (Constant.isReference())
    return buildLoadOfLValue(Constant.getReferenceLValue(*this, E),
                             E->getExprLoc())
        .getScalarVal();
  return builder.getConstant(getLoc(E->getSourceRange()), Constant.getValue());
}

LValue CIRGenFunction::buildPredefinedLValue(const PredefinedExpr *E) {
  const auto *SL = E->getFunctionName();
  assert(SL != nullptr && "No StringLiteral name in PredefinedExpr");
  auto Fn = dyn_cast<mlir::cir::FuncOp>(CurFn);
  assert(Fn && "other callables NYI");
  StringRef FnName = Fn.getName();
  if (FnName.starts_with("\01"))
    FnName = FnName.substr(1);
  StringRef NameItems[] = {PredefinedExpr::getIdentKindName(E->getIdentKind()),
                           FnName};
  std::string GVName = llvm::join(NameItems, NameItems + 2, ".");
  if (auto *BD = dyn_cast_or_null<BlockDecl>(CurCodeDecl)) {
    llvm_unreachable("NYI");
  }

  return buildStringLiteralLValue(SL);
}
