//===-- CIRGenBuilder.h - CIRBuilder implementation  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H
#define LLVM_CLANG_LIB_CIR_CIRGENBUILDER_H

#include "Address.h"
#include "CIRGenRecordLayout.h"
#include "CIRGenTypeCache.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/IR/FPEnv.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <optional>
#include <string>

namespace cir {

class CIRGenFunction;

class CIRGenBuilderTy : public CIRBaseBuilderTy {
  const CIRGenTypeCache &typeCache;
  bool IsFPConstrained = false;
  fp::ExceptionBehavior DefaultConstrainedExcept = fp::ebStrict;
  llvm::RoundingMode DefaultConstrainedRounding = llvm::RoundingMode::Dynamic;

  llvm::StringMap<unsigned> GlobalsVersioning;
  llvm::StringSet<> anonRecordNames;

public:
  CIRGenBuilderTy(mlir::MLIRContext &C, const CIRGenTypeCache &tc)
      : CIRBaseBuilderTy(C), typeCache(tc) {}

  std::string getUniqueAnonRecordName() {
    std::string name = "anon." + std::to_string(anonRecordNames.size());
    anonRecordNames.insert(name);
    return name;
  }

  //
  // Floating point specific helpers
  // -------------------------------
  //

  /// Enable/Disable use of constrained floating point math. When enabled the
  /// CreateF<op>() calls instead create constrained floating point intrinsic
  /// calls. Fast math flags are unaffected by this setting.
  void setIsFPConstrained(bool IsCon) {
    if (IsCon)
      llvm_unreachable("Constrained FP NYI");
    IsFPConstrained = IsCon;
  }

  /// Query for the use of constrained floating point math
  bool getIsFPConstrained() {
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");
    return IsFPConstrained;
  }

  /// Set the exception handling to be used with constrained floating point
  void setDefaultConstrainedExcept(fp::ExceptionBehavior NewExcept) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> ExceptStr =
        convertExceptionBehaviorToStr(NewExcept);
    assert(ExceptStr && "Garbage strict exception behavior!");
#endif
    DefaultConstrainedExcept = NewExcept;
  }

  /// Set the rounding mode handling to be used with constrained floating point
  void setDefaultConstrainedRounding(llvm::RoundingMode NewRounding) {
#ifndef NDEBUG
    std::optional<llvm::StringRef> RoundingStr =
        convertRoundingModeToStr(NewRounding);
    assert(RoundingStr && "Garbage strict rounding mode!");
#endif
    DefaultConstrainedRounding = NewRounding;
  }

  /// Get the exception handling used with constrained floating point
  fp::ExceptionBehavior getDefaultConstrainedExcept() {
    return DefaultConstrainedExcept;
  }

  /// Get the rounding mode handling used with constrained floating point
  llvm::RoundingMode getDefaultConstrainedRounding() {
    return DefaultConstrainedRounding;
  }

  //
  // Attribute helpers
  // -----------------
  //

  /// Get constant address of a global variable as an MLIR attribute.
  /// This wrapper infers the attribute type through the global op.
  mlir::cir::GlobalViewAttr getGlobalViewAttr(mlir::cir::GlobalOp globalOp,
                                              mlir::ArrayAttr indices = {}) {
    auto type = getPointerTo(globalOp.getSymType());
    return getGlobalViewAttr(type, globalOp, indices);
  }

  /// Get constant address of a global variable as an MLIR attribute.
  mlir::cir::GlobalViewAttr getGlobalViewAttr(mlir::cir::PointerType type,
                                              mlir::cir::GlobalOp globalOp,
                                              mlir::ArrayAttr indices = {}) {
    auto symbol = mlir::FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
    return mlir::cir::GlobalViewAttr::get(type, symbol, indices);
  }

  mlir::TypedAttr getConstNullPtrAttr(mlir::Type t) {
    assert(mlir::isa<mlir::cir::PointerType>(t) && "expected cir.ptr");
    return getConstPtrAttr(t, 0);
  }

  mlir::Attribute getString(llvm::StringRef str, mlir::Type eltTy,
                            unsigned size = 0) {
    unsigned finalSize = size ? size : str.size();

    size_t lastNonZeroPos = str.find_last_not_of('\0');
    // If the string is full of null bytes, emit a #cir.zero rather than
    // a #cir.const_array.
    if (lastNonZeroPos == llvm::StringRef::npos) {
      auto arrayTy = mlir::cir::ArrayType::get(getContext(), eltTy, finalSize);
      return getZeroAttr(arrayTy);
    }
    // We will use trailing zeros only if there are more than one zero
    // at the end
    int trailingZerosNum =
        finalSize > lastNonZeroPos + 2 ? finalSize - lastNonZeroPos - 1 : 0;
    auto truncatedArrayTy = mlir::cir::ArrayType::get(
        getContext(), eltTy, finalSize - trailingZerosNum);
    auto fullArrayTy =
        mlir::cir::ArrayType::get(getContext(), eltTy, finalSize);
    return mlir::cir::ConstArrayAttr::get(
        getContext(), fullArrayTy,
        mlir::StringAttr::get(str.drop_back(trailingZerosNum),
                              truncatedArrayTy),
        trailingZerosNum);
  }

  mlir::cir::ConstArrayAttr getConstArray(mlir::Attribute attrs,
                                          mlir::cir::ArrayType arrayTy) {
    return mlir::cir::ConstArrayAttr::get(arrayTy, attrs);
  }

  mlir::Attribute getConstStructOrZeroAttr(mlir::ArrayAttr arrayAttr,
                                           bool packed = false,
                                           mlir::Type type = {}) {
    llvm::SmallVector<mlir::Type, 8> members;
    auto structTy = mlir::dyn_cast<mlir::cir::StructType>(type);
    assert(structTy && "expected cir.struct");

    // Collect members and check if they are all zero.
    bool isZero = true;
    for (auto &attr : arrayAttr) {
      const auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attr);
      members.push_back(typedAttr.getType());
      isZero &= isNullValue(typedAttr);
    }

    // Struct type not specified: create anon struct type from members.
    if (!structTy)
      structTy = getType<mlir::cir::StructType>(members, packed,
                                                mlir::cir::StructType::Struct,
                                                /*ast=*/nullptr);

    // Return zero or anonymous constant struct.
    if (isZero)
      return mlir::cir::ZeroAttr::get(getContext(), structTy);
    return mlir::cir::ConstStructAttr::get(structTy, arrayAttr);
  }

  mlir::cir::ConstStructAttr getAnonConstStruct(mlir::ArrayAttr arrayAttr,
                                                bool packed = false,
                                                mlir::Type ty = {}) {
    llvm::SmallVector<mlir::Type, 4> members;
    for (auto &f : arrayAttr) {
      auto ta = mlir::dyn_cast<mlir::TypedAttr>(f);
      assert(ta && "expected typed attribute member");
      members.push_back(ta.getType());
    }

    if (!ty)
      ty = getAnonStructTy(members, packed);

    auto sTy = mlir::dyn_cast<mlir::cir::StructType>(ty);
    assert(sTy && "expected struct type");
    return mlir::cir::ConstStructAttr::get(sTy, arrayAttr);
  }

  mlir::cir::TypeInfoAttr getTypeInfo(mlir::ArrayAttr fieldsAttr) {
    auto anonStruct = getAnonConstStruct(fieldsAttr);
    return mlir::cir::TypeInfoAttr::get(anonStruct.getType(), fieldsAttr);
  }

  mlir::cir::CmpThreeWayInfoAttr getCmpThreeWayInfoStrongOrdering(
      const llvm::APSInt &lt, const llvm::APSInt &eq, const llvm::APSInt &gt) {
    return mlir::cir::CmpThreeWayInfoAttr::get(
        getContext(), lt.getSExtValue(), eq.getSExtValue(), gt.getSExtValue());
  }

  mlir::cir::CmpThreeWayInfoAttr getCmpThreeWayInfoPartialOrdering(
      const llvm::APSInt &lt, const llvm::APSInt &eq, const llvm::APSInt &gt,
      const llvm::APSInt &unordered) {
    return mlir::cir::CmpThreeWayInfoAttr::get(
        getContext(), lt.getSExtValue(), eq.getSExtValue(), gt.getSExtValue(),
        unordered.getSExtValue());
  }

  mlir::cir::DataMemberAttr getDataMemberAttr(mlir::cir::DataMemberType ty,
                                              unsigned memberIndex) {
    return mlir::cir::DataMemberAttr::get(getContext(), ty, memberIndex);
  }

  mlir::cir::DataMemberAttr
  getNullDataMemberAttr(mlir::cir::DataMemberType ty) {
    return mlir::cir::DataMemberAttr::get(getContext(), ty, std::nullopt);
  }

  mlir::TypedAttr getZeroInitAttr(mlir::Type ty) {
    if (mlir::isa<mlir::cir::IntType>(ty))
      return mlir::cir::IntAttr::get(ty, 0);
    if (auto fltType = mlir::dyn_cast<mlir::cir::SingleType>(ty))
      return mlir::cir::FPAttr::getZero(fltType);
    if (auto fltType = mlir::dyn_cast<mlir::cir::DoubleType>(ty))
      return mlir::cir::FPAttr::getZero(fltType);
    if (auto fltType = mlir::dyn_cast<mlir::cir::FP16Type>(ty))
      return mlir::cir::FPAttr::getZero(fltType);
    if (auto fltType = mlir::dyn_cast<mlir::cir::BF16Type>(ty))
      return mlir::cir::FPAttr::getZero(fltType);
    if (auto complexType = mlir::dyn_cast<mlir::cir::ComplexType>(ty))
      return getZeroAttr(complexType);
    if (auto arrTy = mlir::dyn_cast<mlir::cir::ArrayType>(ty))
      return getZeroAttr(arrTy);
    if (auto ptrTy = mlir::dyn_cast<mlir::cir::PointerType>(ty))
      return getConstNullPtrAttr(ptrTy);
    if (auto structTy = mlir::dyn_cast<mlir::cir::StructType>(ty))
      return getZeroAttr(structTy);
    if (mlir::isa<mlir::cir::BoolType>(ty)) {
      return getCIRBoolAttr(false);
    }
    llvm_unreachable("Zero initializer for given type is NYI");
  }

  // TODO(cir): Once we have CIR float types, replace this by something like a
  // NullableValueInterface to allow for type-independent queries.
  bool isNullValue(mlir::Attribute attr) const {
    if (mlir::isa<mlir::cir::ZeroAttr>(attr))
      return true;
    if (const auto ptrVal = mlir::dyn_cast<mlir::cir::ConstPtrAttr>(attr))
      return ptrVal.isNullValue();

    if (mlir::isa<mlir::cir::GlobalViewAttr>(attr))
      return false;

    // TODO(cir): introduce char type in CIR and check for that instead.
    if (const auto intVal = mlir::dyn_cast<mlir::cir::IntAttr>(attr))
      return intVal.isNullValue();

    if (const auto boolVal = mlir::dyn_cast<mlir::cir::BoolAttr>(attr))
      return !boolVal.getValue();

    if (auto fpAttr = mlir::dyn_cast<mlir::cir::FPAttr>(attr)) {
      auto fpVal = fpAttr.getValue();
      bool ignored;
      llvm::APFloat FV(+0.0);
      FV.convert(fpVal.getSemantics(), llvm::APFloat::rmNearestTiesToEven,
                 &ignored);
      return FV.bitwiseIsEqual(fpVal);
    }

    if (const auto structVal =
            mlir::dyn_cast<mlir::cir::ConstStructAttr>(attr)) {
      for (const auto elt : structVal.getMembers()) {
        // FIXME(cir): the struct's ID should not be considered a member.
        if (mlir::isa<mlir::StringAttr>(elt))
          continue;
        if (!isNullValue(elt))
          return false;
      }
      return true;
    }

    if (const auto arrayVal = mlir::dyn_cast<mlir::cir::ConstArrayAttr>(attr)) {
      if (mlir::isa<mlir::StringAttr>(arrayVal.getElts()))
        return false;
      for (const auto elt : mlir::cast<mlir::ArrayAttr>(arrayVal.getElts())) {
        if (!isNullValue(elt))
          return false;
      }
      return true;
    }

    llvm_unreachable("NYI");
  }

  //
  // Type helpers
  // ------------
  //
  mlir::cir::IntType getUIntNTy(int N) {
    switch (N) {
    case 8:
      return getUInt8Ty();
    case 16:
      return getUInt16Ty();
    case 32:
      return getUInt32Ty();
    case 64:
      return getUInt64Ty();
    default:
      return mlir::cir::IntType::get(getContext(), N, false);
    }
  }

  mlir::cir::IntType getSIntNTy(int N) {
    switch (N) {
    case 8:
      return getSInt8Ty();
    case 16:
      return getSInt16Ty();
    case 32:
      return getSInt32Ty();
    case 64:
      return getSInt64Ty();
    default:
      return mlir::cir::IntType::get(getContext(), N, true);
    }
  }

  mlir::cir::VoidType getVoidTy() { return typeCache.VoidTy; }

  mlir::cir::IntType getSInt8Ty() { return typeCache.SInt8Ty; }
  mlir::cir::IntType getSInt16Ty() { return typeCache.SInt16Ty; }
  mlir::cir::IntType getSInt32Ty() { return typeCache.SInt32Ty; }
  mlir::cir::IntType getSInt64Ty() { return typeCache.SInt64Ty; }

  mlir::cir::IntType getUInt8Ty() { return typeCache.UInt8Ty; }
  mlir::cir::IntType getUInt16Ty() { return typeCache.UInt16Ty; }
  mlir::cir::IntType getUInt32Ty() { return typeCache.UInt32Ty; }
  mlir::cir::IntType getUInt64Ty() { return typeCache.UInt64Ty; }

  bool isInt8Ty(mlir::Type i) {
    return i == typeCache.UInt8Ty || i == typeCache.SInt8Ty;
  }
  bool isInt16Ty(mlir::Type i) {
    return i == typeCache.UInt16Ty || i == typeCache.SInt16Ty;
  }
  bool isInt32Ty(mlir::Type i) {
    return i == typeCache.UInt32Ty || i == typeCache.SInt32Ty;
  }
  bool isInt64Ty(mlir::Type i) {
    return i == typeCache.UInt64Ty || i == typeCache.SInt64Ty;
  }
  bool isInt(mlir::Type i) { return mlir::isa<mlir::cir::IntType>(i); }

  mlir::cir::LongDoubleType
  getLongDoubleTy(const llvm::fltSemantics &format) const {
    if (&format == &llvm::APFloat::IEEEdouble())
      return mlir::cir::LongDoubleType::get(getContext(), typeCache.DoubleTy);
    if (&format == &llvm::APFloat::x87DoubleExtended())
      return mlir::cir::LongDoubleType::get(getContext(), typeCache.FP80Ty);
    if (&format == &llvm::APFloat::IEEEquad())
      llvm_unreachable("NYI");
    if (&format == &llvm::APFloat::PPCDoubleDouble())
      llvm_unreachable("NYI");

    llvm_unreachable("unsupported long double format");
  }

  mlir::Type getVirtualFnPtrType(bool isVarArg = false) {
    // FIXME: replay LLVM codegen for now, perhaps add a vtable ptr special
    // type so it's a bit more clear and C++ idiomatic.
    auto fnTy = mlir::cir::FuncType::get({}, getUInt32Ty(), isVarArg);
    assert(!MissingFeatures::isVarArg());
    return getPointerTo(getPointerTo(fnTy));
  }

  mlir::cir::FuncType getFuncType(llvm::ArrayRef<mlir::Type> params,
                                  mlir::Type retTy, bool isVarArg = false) {
    return mlir::cir::FuncType::get(params, retTy, isVarArg);
  }

  // Fetch the type representing a pointer to unsigned int values.
  mlir::cir::PointerType getUInt8PtrTy(unsigned AddrSpace = 0) {
    return typeCache.UInt8PtrTy;
  }
  mlir::cir::PointerType getUInt32PtrTy(unsigned AddrSpace = 0) {
    return mlir::cir::PointerType::get(getContext(), typeCache.UInt32Ty);
  }

  /// Get a CIR anonymous struct type.
  mlir::cir::StructType
  getAnonStructTy(llvm::ArrayRef<mlir::Type> members, bool packed = false,
                  const clang::RecordDecl *ast = nullptr) {
    mlir::cir::ASTRecordDeclAttr astAttr = nullptr;
    auto kind = mlir::cir::StructType::RecordKind::Struct;
    if (ast) {
      astAttr = getAttr<mlir::cir::ASTRecordDeclAttr>(ast);
      kind = getRecordKind(ast->getTagKind());
    }
    return getType<mlir::cir::StructType>(members, packed, kind, astAttr);
  }

  /// Get a CIR record kind from a AST declaration tag.
  mlir::cir::StructType::RecordKind
  getRecordKind(const clang::TagTypeKind kind) {
    switch (kind) {
    case clang::TagTypeKind::Struct:
      return mlir::cir::StructType::Struct;
    case clang::TagTypeKind::Union:
      return mlir::cir::StructType::Union;
    case clang::TagTypeKind::Class:
      return mlir::cir::StructType::Class;
    case clang::TagTypeKind::Interface:
      llvm_unreachable("interface records are NYI");
    case clang::TagTypeKind::Enum:
      llvm_unreachable("enum records are NYI");
    }
  }

  /// Get a incomplete CIR struct type.
  mlir::cir::StructType getIncompleteStructTy(llvm::StringRef name,
                                              const clang::RecordDecl *ast) {
    const auto nameAttr = getStringAttr(name);
    auto kind = mlir::cir::StructType::RecordKind::Struct;
    if (ast)
      kind = getRecordKind(ast->getTagKind());
    return getType<mlir::cir::StructType>(nameAttr, kind);
  }

  /// Get a CIR named struct type.
  ///
  /// If a struct already exists and is complete, but the client tries to fetch
  /// it with a different set of attributes, this method will crash.
  mlir::cir::StructType getCompleteStructTy(llvm::ArrayRef<mlir::Type> members,
                                            llvm::StringRef name, bool packed,
                                            const clang::RecordDecl *ast) {
    const auto nameAttr = getStringAttr(name);
    mlir::cir::ASTRecordDeclAttr astAttr = nullptr;
    auto kind = mlir::cir::StructType::RecordKind::Struct;
    if (ast) {
      astAttr = getAttr<mlir::cir::ASTRecordDeclAttr>(ast);
      kind = getRecordKind(ast->getTagKind());
    }

    // Create or get the struct.
    auto type = getType<mlir::cir::StructType>(members, nameAttr, packed, kind,
                                               astAttr);

    // Complete an incomplete struct or ensure the existing complete struct
    // matches the requested attributes.
    type.complete(members, packed, astAttr);

    return type;
  }

  mlir::cir::StructType
  getCompleteStructType(mlir::ArrayAttr fields, bool packed = false,
                        llvm::StringRef name = "",
                        const clang::RecordDecl *ast = nullptr) {
    llvm::SmallVector<mlir::Type, 8> members;
    for (auto &attr : fields) {
      const auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attr);
      members.push_back(typedAttr.getType());
    }

    if (name.empty())
      return getAnonStructTy(members, packed, ast);
    else
      return getCompleteStructTy(members, name, packed, ast);
  }

  mlir::cir::ArrayType getArrayType(mlir::Type eltType, unsigned size) {
    return mlir::cir::ArrayType::get(getContext(), eltType, size);
  }

  bool isSized(mlir::Type ty) {
    if (mlir::isa<mlir::cir::PointerType, mlir::cir::StructType,
                  mlir::cir::ArrayType, mlir::cir::BoolType, mlir::cir::IntType,
                  mlir::cir::CIRFPTypeInterface>(ty))
      return true;
    assert(0 && "Unimplemented size for type");
    return false;
  }

  //
  // Constant creation helpers
  // -------------------------
  //
  mlir::cir::ConstantOp getSInt32(uint32_t c, mlir::Location loc) {
    auto sInt32Ty = getSInt32Ty();
    return create<mlir::cir::ConstantOp>(loc, sInt32Ty,
                                         mlir::cir::IntAttr::get(sInt32Ty, c));
  }
  mlir::cir::ConstantOp getUInt32(uint32_t C, mlir::Location loc) {
    auto uInt32Ty = getUInt32Ty();
    return create<mlir::cir::ConstantOp>(loc, uInt32Ty,
                                         mlir::cir::IntAttr::get(uInt32Ty, C));
  }
  mlir::cir::ConstantOp getSInt64(uint64_t C, mlir::Location loc) {
    auto sInt64Ty = getSInt64Ty();
    return create<mlir::cir::ConstantOp>(loc, sInt64Ty,
                                         mlir::cir::IntAttr::get(sInt64Ty, C));
  }
  mlir::cir::ConstantOp getUInt64(uint64_t C, mlir::Location loc) {
    auto uInt64Ty = getUInt64Ty();
    return create<mlir::cir::ConstantOp>(loc, uInt64Ty,
                                         mlir::cir::IntAttr::get(uInt64Ty, C));
  }

  mlir::cir::ConstantOp getConstInt(mlir::Location loc, llvm::APSInt intVal) {
    bool isSigned = intVal.isSigned();
    auto width = intVal.getBitWidth();
    mlir::cir::IntType t = isSigned ? getSIntNTy(width) : getUIntNTy(width);
    return getConstInt(
        loc, t, isSigned ? intVal.getSExtValue() : intVal.getZExtValue());
  }

  mlir::cir::ConstantOp getConstInt(mlir::Location loc, mlir::Type t,
                                    uint64_t C) {
    auto intTy = mlir::dyn_cast<mlir::cir::IntType>(t);
    assert(intTy && "expected mlir::cir::IntType");
    return create<mlir::cir::ConstantOp>(loc, intTy,
                                         mlir::cir::IntAttr::get(t, C));
  }

  mlir::cir::ConstantOp getBool(bool state, mlir::Location loc) {
    return create<mlir::cir::ConstantOp>(loc, getBoolTy(),
                                         getCIRBoolAttr(state));
  }
  mlir::cir::ConstantOp getFalse(mlir::Location loc) {
    return getBool(false, loc);
  }
  mlir::cir::ConstantOp getTrue(mlir::Location loc) {
    return getBool(true, loc);
  }

  /// Create constant nullptr for pointer-to-data-member type ty.
  mlir::cir::ConstantOp getNullDataMemberPtr(mlir::cir::DataMemberType ty,
                                             mlir::Location loc) {
    return create<mlir::cir::ConstantOp>(loc, ty, getNullDataMemberAttr(ty));
  }

  // Creates constant null value for integral type ty.
  mlir::cir::ConstantOp getNullValue(mlir::Type ty, mlir::Location loc) {
    return create<mlir::cir::ConstantOp>(loc, ty, getZeroInitAttr(ty));
  }

  mlir::cir::ConstantOp getZero(mlir::Location loc, mlir::Type ty) {
    // TODO: dispatch creation for primitive types.
    assert((mlir::isa<mlir::cir::StructType>(ty) ||
            mlir::isa<mlir::cir::ArrayType>(ty)) &&
           "NYI for other types");
    return create<mlir::cir::ConstantOp>(loc, ty, getZeroAttr(ty));
  }

  //
  // Operation creation helpers
  // --------------------------
  //

  /// Create a copy with inferred length.
  mlir::cir::CopyOp createCopy(mlir::Value dst, mlir::Value src,
                               bool isVolatile = false) {
    return create<mlir::cir::CopyOp>(dst.getLoc(), dst, src, isVolatile);
  }

  /// Create a break operation.
  mlir::cir::BreakOp createBreak(mlir::Location loc) {
    return create<mlir::cir::BreakOp>(loc);
  }

  /// Create a continue operation.
  mlir::cir::ContinueOp createContinue(mlir::Location loc) {
    return create<mlir::cir::ContinueOp>(loc);
  }

  mlir::cir::MemCpyOp createMemCpy(mlir::Location loc, mlir::Value dst,
                                   mlir::Value src, mlir::Value len) {
    return create<mlir::cir::MemCpyOp>(loc, dst, src, len);
  }

  mlir::Value createNeg(mlir::Value value) {

    if (auto intTy = mlir::dyn_cast<mlir::cir::IntType>(value.getType())) {
      // Source is a unsigned integer: first cast it to signed.
      if (intTy.isUnsigned())
        value = createIntCast(value, getSIntNTy(intTy.getWidth()));
      return create<mlir::cir::UnaryOp>(value.getLoc(), value.getType(),
                                        mlir::cir::UnaryOpKind::Minus, value);
    }

    llvm_unreachable("negation for the given type is NYI");
  }

  // TODO: split this to createFPExt/createFPTrunc when we have dedicated cast
  // operations.
  mlir::Value createFloatingCast(mlir::Value v, mlir::Type destType) {
    if (getIsFPConstrained())
      llvm_unreachable("constrainedfp NYI");

    return create<mlir::cir::CastOp>(v.getLoc(), destType,
                                     mlir::cir::CastKind::floating, v);
  }

  mlir::Value createFSub(mlir::Value lhs, mlir::Value rhs) {
    assert(!MissingFeatures::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!MissingFeatures::foldBinOpFMF());
    return create<mlir::cir::BinOp>(lhs.getLoc(), mlir::cir::BinOpKind::Sub,
                                    lhs, rhs);
  }

  mlir::Value createFAdd(mlir::Value lhs, mlir::Value rhs) {
    assert(!MissingFeatures::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!MissingFeatures::foldBinOpFMF());
    return create<mlir::cir::BinOp>(lhs.getLoc(), mlir::cir::BinOpKind::Add,
                                    lhs, rhs);
  }
  mlir::Value createFMul(mlir::Value lhs, mlir::Value rhs) {
    assert(!MissingFeatures::metaDataNode());
    if (IsFPConstrained)
      llvm_unreachable("Constrained FP NYI");

    assert(!MissingFeatures::foldBinOpFMF());
    return create<mlir::cir::BinOp>(lhs.getLoc(), mlir::cir::BinOpKind::Mul,
                                    lhs, rhs);
  }

  mlir::Value createDynCast(mlir::Location loc, mlir::Value src,
                            mlir::cir::PointerType destType, bool isRefCast,
                            mlir::cir::DynamicCastInfoAttr info) {
    auto castKind = isRefCast ? mlir::cir::DynamicCastKind::ref
                              : mlir::cir::DynamicCastKind::ptr;
    return create<mlir::cir::DynamicCastOp>(loc, destType, castKind, src, info,
                                            /*relative_layout=*/false);
  }

  mlir::Value createDynCastToVoid(mlir::Location loc, mlir::Value src,
                                  bool vtableUseRelativeLayout) {
    // TODO(cir): consider address space here.
    assert(!MissingFeatures::addressSpace());
    auto destTy = getVoidPtrTy();
    return create<mlir::cir::DynamicCastOp>(
        loc, destTy, mlir::cir::DynamicCastKind::ptr, src,
        mlir::cir::DynamicCastInfoAttr{}, vtableUseRelativeLayout);
  }

  cir::Address createBaseClassAddr(mlir::Location loc, cir::Address addr,
                                   mlir::Type destType) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    auto baseAddr =
        create<mlir::cir::BaseClassAddrOp>(loc, ptrTy, addr.getPointer());

    return Address(baseAddr, ptrTy, addr.getAlignment());
  }

  // FIXME(cir): CIRGenBuilder class should have an attribute with a reference
  // to the module so that we don't have search for it or pass it around.
  // FIXME(cir): Track a list of globals, or at least the last one inserted, so
  // that we can insert globals in the same order they are defined by CIRGen.

  [[nodiscard]] mlir::cir::GlobalOp
  createGlobal(mlir::ModuleOp module, mlir::Location loc, mlir::StringRef name,
               mlir::Type type, bool isConst,
               mlir::cir::GlobalLinkageKind linkage) {
    mlir::OpBuilder::InsertionGuard guard(*this);
    setInsertionPointToStart(module.getBody());
    return create<mlir::cir::GlobalOp>(loc, name, type, isConst, linkage);
  }

  /// Creates a versioned global variable. If the symbol is already taken, an ID
  /// will be appended to the symbol. The returned global must always be queried
  /// for its name so it can be referenced correctly.
  [[nodiscard]] mlir::cir::GlobalOp
  createVersionedGlobal(mlir::ModuleOp module, mlir::Location loc,
                        mlir::StringRef name, mlir::Type type, bool isConst,
                        mlir::cir::GlobalLinkageKind linkage) {
    // Create a unique name if the given name is already taken.
    std::string uniqueName;
    if (unsigned version = GlobalsVersioning[name.str()]++)
      uniqueName = name.str() + "." + std::to_string(version);
    else
      uniqueName = name.str();

    return createGlobal(module, loc, uniqueName, type, isConst, linkage);
  }

  mlir::Value createGetGlobal(mlir::cir::GlobalOp global,
                              bool threadLocal = false) {
    return create<mlir::cir::GetGlobalOp>(global.getLoc(),
                                          getPointerTo(global.getSymType()),
                                          global.getName(), threadLocal);
  }

  mlir::Value createGetBitfield(mlir::Location loc, mlir::Type resultType,
                                mlir::Value addr, mlir::Type storageType,
                                const CIRGenBitFieldInfo &info,
                                bool isLvalueVolatile, bool useVolatile) {
    auto offset = useVolatile ? info.VolatileOffset : info.Offset;
    return create<mlir::cir::GetBitfieldOp>(loc, resultType, addr, storageType,
                                            info.Name, info.Size, offset,
                                            info.IsSigned, isLvalueVolatile);
  }

  mlir::Value createSetBitfield(mlir::Location loc, mlir::Type resultType,
                                mlir::Value dstAddr, mlir::Type storageType,
                                mlir::Value src, const CIRGenBitFieldInfo &info,
                                bool isLvalueVolatile, bool useVolatile) {
    auto offset = useVolatile ? info.VolatileOffset : info.Offset;
    return create<mlir::cir::SetBitfieldOp>(
        loc, resultType, dstAddr, storageType, src, info.Name, info.Size,
        offset, info.IsSigned, isLvalueVolatile);
  }

  /// Create a pointer to a record member.
  mlir::Value createGetMember(mlir::Location loc, mlir::Type result,
                              mlir::Value base, llvm::StringRef name,
                              unsigned index) {
    return create<mlir::cir::GetMemberOp>(loc, result, base, name, index);
  }

  /// Create a cir.complex.real_ptr operation that derives a pointer to the real
  /// part of the complex value pointed to by the specified pointer value.
  mlir::Value createRealPtr(mlir::Location loc, mlir::Value value) {
    auto srcPtrTy = mlir::cast<mlir::cir::PointerType>(value.getType());
    auto srcComplexTy =
        mlir::cast<mlir::cir::ComplexType>(srcPtrTy.getPointee());
    return create<mlir::cir::ComplexRealPtrOp>(
        loc, getPointerTo(srcComplexTy.getElementTy()), value);
  }

  Address createRealPtr(mlir::Location loc, Address addr) {
    return Address{createRealPtr(loc, addr.getPointer()), addr.getAlignment()};
  }

  /// Create a cir.complex.imag_ptr operation that derives a pointer to the
  /// imaginary part of the complex value pointed to by the specified pointer
  /// value.
  mlir::Value createImagPtr(mlir::Location loc, mlir::Value value) {
    auto srcPtrTy = mlir::cast<mlir::cir::PointerType>(value.getType());
    auto srcComplexTy =
        mlir::cast<mlir::cir::ComplexType>(srcPtrTy.getPointee());
    return create<mlir::cir::ComplexImagPtrOp>(
        loc, getPointerTo(srcComplexTy.getElementTy()), value);
  }

  Address createImagPtr(mlir::Location loc, Address addr) {
    return Address{createImagPtr(loc, addr.getPointer()), addr.getAlignment()};
  }

  /// Cast the element type of the given address to a different type,
  /// preserving information like the alignment.
  cir::Address createElementBitCast(mlir::Location loc, cir::Address addr,
                                    mlir::Type destType) {
    if (destType == addr.getElementType())
      return addr;

    auto ptrTy = getPointerTo(destType);
    return Address(createBitcast(loc, addr.getPointer(), ptrTy), destType,
                   addr.getAlignment());
  }

  mlir::Value createLoad(mlir::Location loc, Address addr,
                         bool isVolatile = false) {
    auto ptrTy =
        mlir::dyn_cast<mlir::cir::PointerType>(addr.getPointer().getType());
    if (addr.getElementType() != ptrTy.getPointee())
      addr = addr.withPointer(
          createPtrBitcast(addr.getPointer(), addr.getElementType()));

    return create<mlir::cir::LoadOp>(
        loc, addr.getElementType(), addr.getPointer(), /*isDeref=*/false,
        /*is_volatile=*/isVolatile, /*alignment=*/mlir::IntegerAttr{},
        /*mem_order=*/mlir::cir::MemOrderAttr{});
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Type ty,
                                mlir::Value ptr, llvm::MaybeAlign align,
                                bool isVolatile) {
    if (ty != mlir::cast<mlir::cir::PointerType>(ptr.getType()).getPointee())
      ptr = createPtrBitcast(ptr, ty);
    uint64_t alignment = align ? align->value() : 0;
    return CIRBaseBuilderTy::createLoad(loc, ptr, isVolatile, alignment);
  }

  mlir::Value createAlignedLoad(mlir::Location loc, mlir::Type ty,
                                mlir::Value ptr, llvm::MaybeAlign align) {
    // TODO: make sure callsites shouldn't be really passing volatile.
    assert(!MissingFeatures::volatileLoadOrStore());
    return createAlignedLoad(loc, ty, ptr, align, /*isVolatile=*/false);
  }

  mlir::Value
  createAlignedLoad(mlir::Location loc, mlir::Type ty, mlir::Value addr,
                    clang::CharUnits align = clang::CharUnits::One()) {
    return createAlignedLoad(loc, ty, addr, align.getAsAlign());
  }

  mlir::cir::StoreOp createStore(mlir::Location loc, mlir::Value val,
                                 Address dst, bool _volatile = false,
                                 ::mlir::IntegerAttr align = {},
                                 ::mlir::cir::MemOrderAttr order = {}) {
    return CIRBaseBuilderTy::createStore(loc, val, dst.getPointer(), _volatile,
                                         align, order);
  }

  mlir::cir::StoreOp createFlagStore(mlir::Location loc, bool val,
                                     mlir::Value dst) {
    auto flag = getBool(val, loc);
    return CIRBaseBuilderTy::createStore(loc, flag, dst);
  }

  mlir::cir::VecShuffleOp
  createVecShuffle(mlir::Location loc, mlir::Value vec1, mlir::Value vec2,
                   llvm::ArrayRef<mlir::Attribute> maskAttrs) {
    auto vecType = mlir::cast<mlir::cir::VectorType>(vec1.getType());
    auto resultTy = mlir::cir::VectorType::get(
        getContext(), vecType.getEltType(), maskAttrs.size());
    return CIRBaseBuilderTy::create<mlir::cir::VecShuffleOp>(
        loc, resultTy, vec1, vec2, getArrayAttr(maskAttrs));
  }

  mlir::cir::VecShuffleOp createVecShuffle(mlir::Location loc, mlir::Value vec1,
                                           mlir::Value vec2,
                                           llvm::ArrayRef<int64_t> mask) {
    llvm::SmallVector<mlir::Attribute, 4> maskAttrs;
    for (int32_t idx : mask) {
      maskAttrs.push_back(mlir::cir::IntAttr::get(getSInt32Ty(), idx));
    }

    return createVecShuffle(loc, vec1, vec2, maskAttrs);
  }

  mlir::cir::VecShuffleOp createVecShuffle(mlir::Location loc, mlir::Value vec1,
                                           llvm::ArrayRef<int64_t> mask) {
    // FIXME(cir): Support use cir.vec.shuffle with single vec
    // Workaround: pass Vec as both vec1 and vec2
    return createVecShuffle(loc, vec1, vec1, mask);
  }

  mlir::cir::StoreOp
  createAlignedStore(mlir::Location loc, mlir::Value val, mlir::Value dst,
                     clang::CharUnits align = clang::CharUnits::One(),
                     bool _volatile = false,
                     ::mlir::cir::MemOrderAttr order = {}) {
    llvm::MaybeAlign mayAlign = align.getAsAlign();
    mlir::IntegerAttr alignAttr;
    if (mayAlign) {
      uint64_t alignment = mayAlign ? mayAlign->value() : 0;
      alignAttr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(dst.getContext(), 64), alignment);
    }
    return CIRBaseBuilderTy::createStore(loc, val, dst, _volatile, alignAttr,
                                         order);
  }

  // Convert byte offset to sequence of high-level indices suitable for
  // GlobalViewAttr. Ideally we shouldn't deal with low-level offsets at all
  // but currently some parts of Clang AST, which we don't want to touch just
  // yet, return them.
  void computeGlobalViewIndicesFromFlatOffset(
      int64_t Offset, mlir::Type Ty, CIRDataLayout Layout,
      llvm::SmallVectorImpl<int64_t> &Indices) {
    if (!Offset)
      return;

    mlir::Type SubType;

    if (auto ArrayTy = mlir::dyn_cast<mlir::cir::ArrayType>(Ty)) {
      auto EltSize = Layout.getTypeAllocSize(ArrayTy.getEltType());
      Indices.push_back(Offset / EltSize);
      SubType = ArrayTy.getEltType();
      Offset %= EltSize;
    } else if (auto PtrTy = mlir::dyn_cast<mlir::cir::PointerType>(Ty)) {
      auto EltSize = Layout.getTypeAllocSize(PtrTy.getPointee());
      Indices.push_back(Offset / EltSize);
      SubType = PtrTy.getPointee();
      Offset %= EltSize;
    } else if (auto StructTy = mlir::dyn_cast<mlir::cir::StructType>(Ty)) {
      auto Elts = StructTy.getMembers();
      unsigned Pos = 0;
      for (size_t I = 0; I < Elts.size(); ++I) {
        auto EltSize = Layout.getTypeAllocSize(Elts[I]);
        unsigned AlignMask = Layout.getABITypeAlign(Elts[I]) - 1;
        Pos = (Pos + AlignMask) & ~AlignMask;
        if (Offset < Pos + EltSize) {
          Indices.push_back(I);
          SubType = Elts[I];
          Offset -= Pos;
          break;
        }
        Pos += EltSize;
      }
    } else {
      llvm_unreachable("unexpected type");
    }

    assert(SubType);
    computeGlobalViewIndicesFromFlatOffset(Offset, SubType, Layout, Indices);
  }

  mlir::cir::StackSaveOp createStackSave(mlir::Location loc, mlir::Type ty) {
    return create<mlir::cir::StackSaveOp>(loc, ty);
  }

  mlir::cir::StackRestoreOp createStackRestore(mlir::Location loc,
                                               mlir::Value v) {
    return create<mlir::cir::StackRestoreOp>(loc, v);
  }

  // TODO(cir): Change this to hoist alloca to the parent *scope* instead.
  /// Move alloca operation to the parent region.
  void hoistAllocaToParentRegion(mlir::cir::AllocaOp alloca) {
    auto &block = alloca->getParentOp()->getParentRegion()->front();
    const auto allocas = block.getOps<mlir::cir::AllocaOp>();
    if (allocas.empty()) {
      alloca->moveBefore(&block, block.begin());
    } else {
      alloca->moveAfter(*std::prev(allocas.end()));
    }
  }

  mlir::cir::CmpThreeWayOp
  createThreeWayCmpStrong(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                          const llvm::APSInt &ltRes, const llvm::APSInt &eqRes,
                          const llvm::APSInt &gtRes) {
    assert(ltRes.getBitWidth() == eqRes.getBitWidth() &&
           ltRes.getBitWidth() == gtRes.getBitWidth() &&
           "the three comparison results must have the same bit width");
    auto cmpResultTy = getSIntNTy(ltRes.getBitWidth());
    auto infoAttr = getCmpThreeWayInfoStrongOrdering(ltRes, eqRes, gtRes);
    return create<mlir::cir::CmpThreeWayOp>(loc, cmpResultTy, lhs, rhs,
                                            infoAttr);
  }

  mlir::cir::CmpThreeWayOp
  createThreeWayCmpPartial(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                           const llvm::APSInt &ltRes, const llvm::APSInt &eqRes,
                           const llvm::APSInt &gtRes,
                           const llvm::APSInt &unorderedRes) {
    assert(ltRes.getBitWidth() == eqRes.getBitWidth() &&
           ltRes.getBitWidth() == gtRes.getBitWidth() &&
           ltRes.getBitWidth() == unorderedRes.getBitWidth() &&
           "the four comparison results must have the same bit width");
    auto cmpResultTy = getSIntNTy(ltRes.getBitWidth());
    auto infoAttr =
        getCmpThreeWayInfoPartialOrdering(ltRes, eqRes, gtRes, unorderedRes);
    return create<mlir::cir::CmpThreeWayOp>(loc, cmpResultTy, lhs, rhs,
                                            infoAttr);
  }

  mlir::cir::GetRuntimeMemberOp createGetIndirectMember(mlir::Location loc,
                                                        mlir::Value objectPtr,
                                                        mlir::Value memberPtr) {
    auto memberPtrTy =
        mlir::cast<mlir::cir::DataMemberType>(memberPtr.getType());

    // TODO(cir): consider address space.
    assert(!MissingFeatures::addressSpace());
    auto resultTy = getPointerTo(memberPtrTy.getMemberTy());

    return create<mlir::cir::GetRuntimeMemberOp>(loc, resultTy, objectPtr,
                                                 memberPtr);
  }
};

} // namespace cir
#endif
