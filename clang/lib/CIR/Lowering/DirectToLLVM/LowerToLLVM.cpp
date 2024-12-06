//====- LowerToLLVM.cpp - Lowering from CIR to LLVMIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to LLVMIR.
//
//===----------------------------------------------------------------------===//
#include "LowerToLLVM.h"
#include "LoweringHelpers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <cstdint>
#include <deque>
#include <optional>
#include <set>

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

namespace {

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(mlir::Region &region,
                        mlir::function_ref<void(mlir::Operation *)> callback) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<Ops...>(op))
      return mlir::WalkResult::skip();
    callback(op);
    return mlir::WalkResult::advance();
  });
}

/// Convert from a CIR comparison kind to an LLVM IR integral comparison kind.
mlir::LLVM::ICmpPredicate convertCmpKindToICmpPredicate(cir::CmpOpKind kind,
                                                        bool isSigned) {
  using CIR = cir::CmpOpKind;
  using LLVMICmp = mlir::LLVM::ICmpPredicate;
  switch (kind) {
  case CIR::eq:
    return LLVMICmp::eq;
  case CIR::ne:
    return LLVMICmp::ne;
  case CIR::lt:
    return (isSigned ? LLVMICmp::slt : LLVMICmp::ult);
  case CIR::le:
    return (isSigned ? LLVMICmp::sle : LLVMICmp::ule);
  case CIR::gt:
    return (isSigned ? LLVMICmp::sgt : LLVMICmp::ugt);
  case CIR::ge:
    return (isSigned ? LLVMICmp::sge : LLVMICmp::uge);
  }
  llvm_unreachable("Unknown CmpOpKind");
}

/// Convert from a CIR comparison kind to an LLVM IR floating-point comparison
/// kind.
mlir::LLVM::FCmpPredicate convertCmpKindToFCmpPredicate(cir::CmpOpKind kind) {
  using CIR = cir::CmpOpKind;
  using LLVMFCmp = mlir::LLVM::FCmpPredicate;
  switch (kind) {
  case CIR::eq:
    return LLVMFCmp::oeq;
  case CIR::ne:
    return LLVMFCmp::une;
  case CIR::lt:
    return LLVMFCmp::olt;
  case CIR::le:
    return LLVMFCmp::ole;
  case CIR::gt:
    return LLVMFCmp::ogt;
  case CIR::ge:
    return LLVMFCmp::oge;
  }
  llvm_unreachable("Unknown CmpOpKind");
}

/// If the given type is a vector type, return the vector's element type.
/// Otherwise return the given type unchanged.
mlir::Type elementTypeIfVector(mlir::Type type) {
  if (auto VecType = mlir::dyn_cast<cir::VectorType>(type)) {
    return VecType.getEltType();
  }
  return type;
}

mlir::LLVM::Visibility
lowerCIRVisibilityToLLVMVisibility(cir::VisibilityKind visibilityKind) {
  switch (visibilityKind) {
  case cir::VisibilityKind::Default:
    return ::mlir::LLVM::Visibility::Default;
  case cir::VisibilityKind::Hidden:
    return ::mlir::LLVM::Visibility::Hidden;
  case cir::VisibilityKind::Protected:
    return ::mlir::LLVM::Visibility::Protected;
  }
}

// Make sure the LLVM function we are about to create a call for actually
// exists, if not create one. Returns a function
void getOrCreateLLVMFuncOp(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Operation *srcOp, llvm::StringRef fnName,
                           mlir::Type fnTy) {
  auto modOp = srcOp->getParentOfType<mlir::ModuleOp>();
  auto enclosingFnOp = srcOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  auto *sourceSymbol = mlir::SymbolTable::lookupSymbolIn(modOp, fnName);
  if (!sourceSymbol) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(enclosingFnOp);
    rewriter.create<mlir::LLVM::LLVMFuncOp>(srcOp->getLoc(), fnName, fnTy);
  }
}

static constexpr StringRef llvmMetadataSectionName = "llvm.metadata";

// Create a string global for annotation related string.
mlir::LLVM::GlobalOp
getAnnotationStringGlobal(mlir::StringAttr strAttr, mlir::ModuleOp &module,
                          llvm::StringMap<mlir::LLVM::GlobalOp> &globalsMap,
                          mlir::OpBuilder &globalVarBuilder,
                          mlir::Location &loc, bool isArg = false) {
  llvm::StringRef str = strAttr.getValue();
  if (!globalsMap.contains(str)) {
    auto llvmStrTy = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(module.getContext(), 8), str.size() + 1);
    auto strGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
        loc, llvmStrTy,
        /*isConstant=*/true, mlir::LLVM::Linkage::Private,
        ".str" +
            (globalsMap.empty() ? ""
                                : "." + std::to_string(globalsMap.size())) +
            ".annotation" + (isArg ? ".arg" : ""),
        mlir::StringAttr::get(module.getContext(), std::string(str) + '\0'),
        /*alignment=*/isArg ? 1 : 0);
    if (!isArg)
      strGlobalOp.setSection(llvmMetadataSectionName);
    strGlobalOp.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
    strGlobalOp.setDsoLocal(true);
    globalsMap[str] = strGlobalOp;
  }
  return globalsMap[str];
}

mlir::LLVM::GlobalOp getOrCreateAnnotationArgsVar(
    mlir::Location &loc, mlir::ModuleOp &module,
    mlir::OpBuilder &globalVarBuilder,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap,
    mlir::ArrayAttr argsAttr) {
  if (argsVarMap.contains(argsAttr))
    return argsVarMap[argsAttr];

  mlir::LLVM::LLVMPointerType annoPtrTy =
      mlir::LLVM::LLVMPointerType::get(globalVarBuilder.getContext());
  llvm::SmallVector<mlir::Type> argStrutFldTypes;
  llvm::SmallVector<mlir::Value> argStrutFields;
  for (mlir::Attribute arg : argsAttr) {
    if (auto strArgAttr = mlir::dyn_cast<mlir::StringAttr>(arg)) {
      // Call getAnnotationStringGlobal here to make sure
      // have a global for this string before
      // creation of the args var.
      getAnnotationStringGlobal(strArgAttr, module, argStringGlobalsMap,
                                globalVarBuilder, loc, true);
      // This will become a ptr to the global string
      argStrutFldTypes.push_back(annoPtrTy);
    } else if (auto intArgAttr = mlir::dyn_cast<mlir::IntegerAttr>(arg)) {
      argStrutFldTypes.push_back(intArgAttr.getType());
    } else {
      llvm_unreachable("Unsupported annotation arg type");
    }
  }

  mlir::LLVM::LLVMStructType argsStructTy =
      mlir::LLVM::LLVMStructType::getLiteral(globalVarBuilder.getContext(),
                                             argStrutFldTypes);
  auto argsGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
      loc, argsStructTy, true, mlir::LLVM::Linkage::Private,
      ".args" +
          (argsVarMap.empty() ? "" : "." + std::to_string(argsVarMap.size())) +
          ".annotation",
      mlir::Attribute());
  argsGlobalOp.setSection(llvmMetadataSectionName);
  argsGlobalOp.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
  argsGlobalOp.setDsoLocal(true);

  // Create the initializer for this args global
  argsGlobalOp.getRegion().push_back(new mlir::Block());
  mlir::OpBuilder argsInitBuilder(module.getContext());
  argsInitBuilder.setInsertionPointToEnd(argsGlobalOp.getInitializerBlock());

  mlir::Value argsStructInit =
      argsInitBuilder.create<mlir::LLVM::UndefOp>(loc, argsStructTy);
  int idx = 0;
  for (mlir::Attribute arg : argsAttr) {
    if (auto strArgAttr = mlir::dyn_cast<mlir::StringAttr>(arg)) {
      // This would be simply return with existing map entry value
      // from argStringGlobalsMap as string global is already
      // created in the previous loop.
      mlir::LLVM::GlobalOp argStrVar = getAnnotationStringGlobal(
          strArgAttr, module, argStringGlobalsMap, globalVarBuilder, loc, true);
      auto argStrVarAddr = argsInitBuilder.create<mlir::LLVM::AddressOfOp>(
          loc, annoPtrTy, argStrVar.getSymName());
      argsStructInit = argsInitBuilder.create<mlir::LLVM::InsertValueOp>(
          loc, argsStructInit, argStrVarAddr, idx++);
    } else if (auto intArgAttr = mlir::dyn_cast<mlir::IntegerAttr>(arg)) {
      auto intArgFld = argsInitBuilder.create<mlir::LLVM::ConstantOp>(
          loc, intArgAttr.getType(), intArgAttr.getValue());
      argsStructInit = argsInitBuilder.create<mlir::LLVM::InsertValueOp>(
          loc, argsStructInit, intArgFld, idx++);
    } else {
      llvm_unreachable("Unsupported annotation arg type");
    }
  }
  argsInitBuilder.create<mlir::LLVM::ReturnOp>(loc, argsStructInit);
  argsVarMap[argsAttr] = argsGlobalOp;
  return argsGlobalOp;
}

/// Lower an annotation value to a series of LLVM globals, `outVals` contains
/// all values which are either used to build other globals or for intrisic call
/// arguments.
void lowerAnnotationValue(
    mlir::Location &localLoc, mlir::Location annotLoc,
    cir::AnnotationAttr annotation, mlir::ModuleOp &module,
    mlir::OpBuilder &varInitBuilder, mlir::OpBuilder &globalVarBuilder,
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap,
    SmallVectorImpl<mlir::Value> &outVals) {
  mlir::LLVM::LLVMPointerType annoPtrTy =
      mlir::LLVM::LLVMPointerType::get(globalVarBuilder.getContext());
  // First field is either a global name or a alloca address and is handled
  // by the caller, this function deals with content from `AnnotationAttr`
  // only.

  // The second field is ptr to the annotation name
  mlir::StringAttr annotationName = annotation.getName();
  auto annotationNameFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
      localLoc, annoPtrTy,
      getAnnotationStringGlobal(annotationName, module, stringGlobalsMap,
                                globalVarBuilder, localLoc)
          .getSymName());
  outVals.push_back(annotationNameFld->getResult(0));

  // The third field is ptr to the translation unit name,
  // and the fourth field is the line number
  if (mlir::isa<mlir::FusedLoc>(annotLoc)) {
    auto FusedLoc = mlir::cast<mlir::FusedLoc>(annotLoc);
    annotLoc = FusedLoc.getLocations()[0];
  }
  auto annotFileLoc = mlir::cast<mlir::FileLineColLoc>(annotLoc);
  assert(annotFileLoc && "annotation value has to be FileLineColLoc");
  // To be consistent with clang code gen, we add trailing null char
  auto fileName = mlir::StringAttr::get(
      module.getContext(), std::string(annotFileLoc.getFilename().getValue()));
  auto fileNameFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
      localLoc, annoPtrTy,
      getAnnotationStringGlobal(fileName, module, stringGlobalsMap,
                                globalVarBuilder, localLoc)
          .getSymName());
  outVals.push_back(fileNameFld->getResult(0));

  unsigned int lineNo = annotFileLoc.getLine();
  auto lineNoFld = varInitBuilder.create<mlir::LLVM::ConstantOp>(
      localLoc, globalVarBuilder.getI32Type(), lineNo);
  outVals.push_back(lineNoFld->getResult(0));

  // The fifth field is ptr to the annotation args var, it could be null
  if (annotation.isNoArgs()) {
    auto nullPtrFld =
        varInitBuilder.create<mlir::LLVM::ZeroOp>(localLoc, annoPtrTy);
    outVals.push_back(nullPtrFld->getResult(0));
  } else {
    mlir::ArrayAttr argsAttr = annotation.getArgs();
    mlir::LLVM::GlobalOp annotArgsVar =
        getOrCreateAnnotationArgsVar(localLoc, module, globalVarBuilder,
                                     argStringGlobalsMap, argsVarMap, argsAttr);
    auto argsVarView = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
        localLoc, annoPtrTy, annotArgsVar.getSymName());
    outVals.push_back(argsVarView->getResult(0));
  }
}

// Get addrspace by converting a pointer type.
// TODO: The approach here is a little hacky. We should access the target info
// directly to convert the address space of global op, similar to what we do
// for type converter.
unsigned getGlobalOpTargetAddrSpace(mlir::ConversionPatternRewriter &rewriter,
                                    const mlir::TypeConverter *converter,
                                    cir::GlobalOp op) {
  auto tempPtrTy = cir::PointerType::get(rewriter.getContext(), op.getSymType(),
                                         op.getAddrSpaceAttr());
  return cast<mlir::LLVM::LLVMPointerType>(converter->convertType(tempPtrTy))
      .getAddressSpace();
}

} // namespace

//===----------------------------------------------------------------------===//
// Visitors for Lowering CIR Const Attributes
//===----------------------------------------------------------------------===//

/// Switches on the type of attribute and calls the appropriate conversion.

/// IntAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::IntAttr intAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(intAttr.getType()), intAttr.getValue());
}

/// BoolAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::BoolAttr boolAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(boolAttr.getType()), boolAttr.getValue());
}

/// ConstPtrAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::ConstPtrAttr ptrAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  if (ptrAttr.isNullValue()) {
    return rewriter.create<mlir::LLVM::ZeroOp>(
        loc, converter->convertType(ptrAttr.getType()));
  }
  mlir::DataLayout layout(parentOp->getParentOfType<mlir::ModuleOp>());
  mlir::Value ptrVal = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getIntegerType(layout.getTypeSizeInBits(ptrAttr.getType())),
      ptrAttr.getValue().getInt());
  return rewriter.create<mlir::LLVM::IntToPtrOp>(
      loc, converter->convertType(ptrAttr.getType()), ptrVal);
}

/// FPAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::FPAttr fltAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, converter->convertType(fltAttr.getType()), fltAttr.getValue());
}

/// ZeroAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::ZeroAttr zeroAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::ZeroOp>(
      loc, converter->convertType(zeroAttr.getType()));
}

/// UndefAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::UndefAttr undefAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto loc = parentOp->getLoc();
  return rewriter.create<mlir::LLVM::UndefOp>(
      loc, converter->convertType(undefAttr.getType()));
}

/// ConstStruct visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::ConstStructAttr constStruct,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constStruct.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  // Iteratively lower each constant element of the struct.
  for (auto [idx, elt] : llvm::enumerate(constStruct.getMembers())) {
    mlir::Value init = lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// VTableAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::VTableAttr vtableArr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(vtableArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(vtableArr.getVtableData())) {
    mlir::Value init = lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// TypeInfoAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::TypeInfoAttr typeinfoArr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(typeinfoArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);

  for (auto [idx, elt] : llvm::enumerate(typeinfoArr.getData())) {
    mlir::Value init = lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
  }

  return result;
}

// ConstArrayAttr visitor
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::ConstArrayAttr constArr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constArr.getType());
  auto loc = parentOp->getLoc();
  mlir::Value result;

  if (auto zeros = constArr.getTrailingZerosNum()) {
    auto arrayTy = constArr.getType();
    result = rewriter.create<mlir::LLVM::ZeroOp>(
        loc, converter->convertType(arrayTy));
  } else {
    result = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmTy);
  }

  // Iteratively lower each constant element of the array.
  if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts())) {
    for (auto [idx, elt] : llvm::enumerate(arrayAttr)) {
      mlir::Value init =
          lowerCirAttrAsValue(parentOp, elt, rewriter, converter);
      result =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
    }
  }
  // TODO(cir): this diverges from traditional lowering. Normally the string
  // would be a global constant that is memcopied.
  else if (auto strAttr =
               mlir::dyn_cast<mlir::StringAttr>(constArr.getElts())) {
    auto arrayTy = mlir::dyn_cast<cir::ArrayType>(strAttr.getType());
    assert(arrayTy && "String attribute must have an array type");
    auto eltTy = arrayTy.getEltType();
    for (auto [idx, elt] : llvm::enumerate(strAttr)) {
      auto init = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, converter->convertType(eltTy), elt);
      result =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, result, init, idx);
    }
  } else {
    llvm_unreachable("unexpected ConstArrayAttr elements");
  }

  return result;
}

// ConstVectorAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::ConstVectorAttr constVec,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto llvmTy = converter->convertType(constVec.getType());
  auto loc = parentOp->getLoc();
  SmallVector<mlir::Attribute> mlirValues;
  for (auto elementAttr : constVec.getElts()) {
    mlir::Attribute mlirAttr;
    if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(elementAttr)) {
      mlirAttr = rewriter.getIntegerAttr(
          converter->convertType(intAttr.getType()), intAttr.getValue());
    } else if (auto floatAttr = mlir::dyn_cast<cir::FPAttr>(elementAttr)) {
      mlirAttr = rewriter.getFloatAttr(
          converter->convertType(floatAttr.getType()), floatAttr.getValue());
    } else {
      llvm_unreachable(
          "vector constant with an element that is neither an int nor a float");
    }
    mlirValues.push_back(mlirAttr);
  }
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, llvmTy,
      mlir::DenseElementsAttr::get(mlir::cast<mlir::ShapedType>(llvmTy),
                                   mlirValues));
}

// GlobalViewAttr visitor.
static mlir::Value
lowerCirAttrAsValue(mlir::Operation *parentOp, cir::GlobalViewAttr globalAttr,
                    mlir::ConversionPatternRewriter &rewriter,
                    const mlir::TypeConverter *converter) {
  auto module = parentOp->getParentOfType<mlir::ModuleOp>();
  mlir::Type sourceType;
  unsigned sourceAddrSpace = 0;
  llvm::StringRef symName;
  auto *sourceSymbol =
      mlir::SymbolTable::lookupSymbolIn(module, globalAttr.getSymbol());
  if (auto llvmSymbol = dyn_cast<mlir::LLVM::GlobalOp>(sourceSymbol)) {
    sourceType = llvmSymbol.getType();
    symName = llvmSymbol.getSymName();
    sourceAddrSpace = llvmSymbol.getAddrSpace();
  } else if (auto cirSymbol = dyn_cast<cir::GlobalOp>(sourceSymbol)) {
    sourceType = converter->convertType(cirSymbol.getSymType());
    symName = cirSymbol.getSymName();
    sourceAddrSpace =
        getGlobalOpTargetAddrSpace(rewriter, converter, cirSymbol);
  } else if (auto llvmFun = dyn_cast<mlir::LLVM::LLVMFuncOp>(sourceSymbol)) {
    sourceType = llvmFun.getFunctionType();
    symName = llvmFun.getSymName();
    sourceAddrSpace = 0;
  } else if (auto fun = dyn_cast<cir::FuncOp>(sourceSymbol)) {
    sourceType = converter->convertType(fun.getFunctionType());
    symName = fun.getSymName();
    sourceAddrSpace = 0;
  } else {
    llvm_unreachable("Unexpected GlobalOp type");
  }

  auto loc = parentOp->getLoc();
  mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
      loc,
      mlir::LLVM::LLVMPointerType::get(rewriter.getContext(), sourceAddrSpace),
      symName);

  if (globalAttr.getIndices()) {
    llvm::SmallVector<mlir::LLVM::GEPArg> indices;

    if (auto stTy = dyn_cast<mlir::LLVM::LLVMStructType>(sourceType)) {
      if (stTy.isIdentified())
        indices.push_back(0);
    } else if (isa<mlir::LLVM::LLVMArrayType>(sourceType)) {
      indices.push_back(0);
    }

    for (auto idx : globalAttr.getIndices()) {
      auto intAttr = dyn_cast<mlir::IntegerAttr>(idx);
      assert(intAttr && "index must be integers");
      indices.push_back(intAttr.getValue().getSExtValue());
    }
    auto resTy = addrOp.getType();
    auto eltTy = converter->convertType(sourceType);
    addrOp = rewriter.create<mlir::LLVM::GEPOp>(loc, resTy, eltTy, addrOp,
                                                indices, true);
  }

  auto ptrTy = mlir::dyn_cast<cir::PointerType>(globalAttr.getType());
  assert(ptrTy && "Expecting pointer type in GlobalViewAttr");
  auto llvmEltTy = converter->convertType(ptrTy.getPointee());

  if (llvmEltTy == sourceType)
    return addrOp;

  auto llvmDstTy = converter->convertType(globalAttr.getType());
  return rewriter.create<mlir::LLVM::BitcastOp>(parentOp->getLoc(), llvmDstTy,
                                                addrOp);
}

/// Switches on the type of attribute and calls the appropriate conversion.
mlir::Value lowerCirAttrAsValue(mlir::Operation *parentOp, mlir::Attribute attr,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *converter) {
  if (const auto intAttr = mlir::dyn_cast<cir::IntAttr>(attr))
    return lowerCirAttrAsValue(parentOp, intAttr, rewriter, converter);
  if (const auto fltAttr = mlir::dyn_cast<cir::FPAttr>(attr))
    return lowerCirAttrAsValue(parentOp, fltAttr, rewriter, converter);
  if (const auto ptrAttr = mlir::dyn_cast<cir::ConstPtrAttr>(attr))
    return lowerCirAttrAsValue(parentOp, ptrAttr, rewriter, converter);
  if (const auto constStruct = mlir::dyn_cast<cir::ConstStructAttr>(attr))
    return lowerCirAttrAsValue(parentOp, constStruct, rewriter, converter);
  if (const auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(attr))
    return lowerCirAttrAsValue(parentOp, constArr, rewriter, converter);
  if (const auto constVec = mlir::dyn_cast<cir::ConstVectorAttr>(attr))
    return lowerCirAttrAsValue(parentOp, constVec, rewriter, converter);
  if (const auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(attr))
    return lowerCirAttrAsValue(parentOp, boolAttr, rewriter, converter);
  if (const auto zeroAttr = mlir::dyn_cast<cir::ZeroAttr>(attr))
    return lowerCirAttrAsValue(parentOp, zeroAttr, rewriter, converter);
  if (const auto undefAttr = mlir::dyn_cast<cir::UndefAttr>(attr))
    return lowerCirAttrAsValue(parentOp, undefAttr, rewriter, converter);
  if (const auto globalAttr = mlir::dyn_cast<cir::GlobalViewAttr>(attr))
    return lowerCirAttrAsValue(parentOp, globalAttr, rewriter, converter);
  if (const auto vtableAttr = mlir::dyn_cast<cir::VTableAttr>(attr))
    return lowerCirAttrAsValue(parentOp, vtableAttr, rewriter, converter);
  if (const auto typeinfoAttr = mlir::dyn_cast<cir::TypeInfoAttr>(attr))
    return lowerCirAttrAsValue(parentOp, typeinfoAttr, rewriter, converter);

  llvm_unreachable("unhandled attribute type");
}

//===----------------------------------------------------------------------===//

mlir::LLVM::Linkage convertLinkage(cir::GlobalLinkageKind linkage) {
  using CIR = cir::GlobalLinkageKind;
  using LLVM = mlir::LLVM::Linkage;

  switch (linkage) {
  case CIR::AvailableExternallyLinkage:
    return LLVM::AvailableExternally;
  case CIR::CommonLinkage:
    return LLVM::Common;
  case CIR::ExternalLinkage:
    return LLVM::External;
  case CIR::ExternalWeakLinkage:
    return LLVM::ExternWeak;
  case CIR::InternalLinkage:
    return LLVM::Internal;
  case CIR::LinkOnceAnyLinkage:
    return LLVM::Linkonce;
  case CIR::LinkOnceODRLinkage:
    return LLVM::LinkonceODR;
  case CIR::PrivateLinkage:
    return LLVM::Private;
  case CIR::WeakAnyLinkage:
    return LLVM::Weak;
  case CIR::WeakODRLinkage:
    return LLVM::WeakODR;
  };
}

mlir::LLVM::CConv convertCallingConv(cir::CallingConv callinvConv) {
  using CIR = cir::CallingConv;
  using LLVM = mlir::LLVM::CConv;

  switch (callinvConv) {
  case CIR::C:
    return LLVM::C;
  case CIR::SpirKernel:
    return LLVM::SPIR_KERNEL;
  case CIR::SpirFunction:
    return LLVM::SPIR_FUNC;
  }
  llvm_unreachable("Unknown calling convention");
}

mlir::LogicalResult CIRToLLVMCopyOpLowering::matchAndRewrite(
    cir::CopyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const mlir::Value length = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI32Type(), op.getLength());
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyOp>(
      op, adaptor.getDst(), adaptor.getSrc(), length, op.getIsVolatile());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemCpyOpLowering::matchAndRewrite(
    cir::MemCpyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
      /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemChrOpLowering::matchAndRewrite(
    cir::MemChrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  llvm::SmallVector<mlir::Type> arguments;
  const mlir::TypeConverter *converter = getTypeConverter();
  mlir::Type srcTy = converter->convertType(op.getSrc().getType());
  mlir::Type patternTy = converter->convertType(op.getPattern().getType());
  mlir::Type lenTy = converter->convertType(op.getLen().getType());
  auto fnTy =
      mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {srcTy, patternTy, lenTy},
                                        /*isVarArg=*/false);
  llvm::StringRef fnName = "memchr";
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{llvmPtrTy}, fnName,
      mlir::ValueRange{adaptor.getSrc(), adaptor.getPattern(),
                       adaptor.getLen()});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemMoveOpLowering::matchAndRewrite(
    cir::MemMoveOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemmoveOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLen(),
      /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemCpyInlineOpLowering::matchAndRewrite(
    cir::MemCpyInlineOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemcpyInlineOp>(
      op, adaptor.getDst(), adaptor.getSrc(), adaptor.getLenAttr(),
      /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemSetOpLowering::matchAndRewrite(
    cir::MemSetOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto converted = rewriter.create<mlir::LLVM::TruncOp>(
      op.getLoc(), mlir::IntegerType::get(op.getContext(), 8),
      adaptor.getVal());
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemsetOp>(op, adaptor.getDst(),
                                                    converted, adaptor.getLen(),
                                                    /*isVolatile=*/false);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMMemSetInlineOpLowering::matchAndRewrite(
    cir::MemSetInlineOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto converted = rewriter.create<mlir::LLVM::TruncOp>(
      op.getLoc(), mlir::IntegerType::get(op.getContext(), 8),
      adaptor.getVal());
  rewriter.replaceOpWithNewOp<mlir::LLVM::MemsetInlineOp>(
      op, adaptor.getDst(), converted, adaptor.getLenAttr(),
      /*isVolatile=*/false);
  return mlir::success();
}

static mlir::Value getLLVMIntCast(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Value llvmSrc, mlir::Type llvmDstIntTy,
                                  bool isUnsigned, uint64_t cirSrcWidth,
                                  uint64_t cirDstIntWidth) {
  if (cirSrcWidth == cirDstIntWidth)
    return llvmSrc;

  auto loc = llvmSrc.getLoc();
  if (cirSrcWidth < cirDstIntWidth) {
    if (isUnsigned)
      return rewriter.create<mlir::LLVM::ZExtOp>(loc, llvmDstIntTy, llvmSrc);
    return rewriter.create<mlir::LLVM::SExtOp>(loc, llvmDstIntTy, llvmSrc);
  }

  // Otherwise truncate
  return rewriter.create<mlir::LLVM::TruncOp>(loc, llvmDstIntTy, llvmSrc);
}

mlir::LogicalResult CIRToLLVMPtrStrideOpLowering::matchAndRewrite(
    cir::PtrStrideOp ptrStrideOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto *tc = getTypeConverter();
  const auto resultTy = tc->convertType(ptrStrideOp.getType());
  auto elementTy = tc->convertType(ptrStrideOp.getElementTy());
  auto *ctx = elementTy.getContext();

  // void and function types doesn't really have a layout to use in GEPs,
  // make it i8 instead.
  if (mlir::isa<mlir::LLVM::LLVMVoidType>(elementTy) ||
      mlir::isa<mlir::LLVM::LLVMFunctionType>(elementTy))
    elementTy = mlir::IntegerType::get(elementTy.getContext(), 8,
                                       mlir::IntegerType::Signless);

  // Zero-extend, sign-extend or trunc the pointer value.
  auto index = adaptor.getStride();
  auto width = mlir::cast<mlir::IntegerType>(index.getType()).getWidth();
  mlir::DataLayout LLVMLayout(ptrStrideOp->getParentOfType<mlir::ModuleOp>());
  auto layoutWidth =
      LLVMLayout.getTypeIndexBitwidth(adaptor.getBase().getType());
  auto indexOp = index.getDefiningOp();
  if (indexOp && layoutWidth && width != *layoutWidth) {
    // If the index comes from a subtraction, make sure the extension happens
    // before it. To achieve that, look at unary minus, which already got
    // lowered to "sub 0, x".
    auto sub = dyn_cast<mlir::LLVM::SubOp>(indexOp);
    auto unary = dyn_cast_if_present<cir::UnaryOp>(
        ptrStrideOp.getStride().getDefiningOp());
    bool rewriteSub =
        unary && unary.getKind() == cir::UnaryOpKind::Minus && sub;
    if (rewriteSub)
      index = indexOp->getOperand(1);

    // Handle the cast
    auto llvmDstType = mlir::IntegerType::get(ctx, *layoutWidth);
    index = getLLVMIntCast(rewriter, index, llvmDstType,
                           ptrStrideOp.getStride().getType().isUnsigned(),
                           width, *layoutWidth);

    // Rewrite the sub in front of extensions/trunc
    if (rewriteSub) {
      index = rewriter.create<mlir::LLVM::SubOp>(
          index.getLoc(), index.getType(),
          rewriter.create<mlir::LLVM::ConstantOp>(
              index.getLoc(), index.getType(),
              mlir::IntegerAttr::get(index.getType(), 0)),
          index);
      rewriter.eraseOp(sub);
    }
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      ptrStrideOp, resultTy, elementTy, adaptor.getBase(), index);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBaseClassAddrOpLowering::matchAndRewrite(
    cir::BaseClassAddrOp baseClassOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto resultType =
      getTypeConverter()->convertType(baseClassOp.getType());
  mlir::Value derivedAddr = adaptor.getDerivedAddr();
  llvm::SmallVector<mlir::LLVM::GEPArg, 1> offset = {
      adaptor.getOffset().getZExtValue()};
  mlir::Type byteType = mlir::IntegerType::get(resultType.getContext(), 8,
                                               mlir::IntegerType::Signless);
  if (adaptor.getOffset().getZExtValue() == 0) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(
        baseClassOp, resultType, adaptor.getDerivedAddr());
    return mlir::success();
  }

  if (baseClassOp.getAssumeNotNull()) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        baseClassOp, resultType, byteType, derivedAddr, offset);
  } else {
    auto loc = baseClassOp.getLoc();
    mlir::Value isNull = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, derivedAddr,
        rewriter.create<mlir::LLVM::ZeroOp>(loc, derivedAddr.getType()));
    mlir::Value adjusted = rewriter.create<mlir::LLVM::GEPOp>(
        loc, resultType, byteType, derivedAddr, offset);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(baseClassOp, isNull,
                                                      derivedAddr, adjusted);
  }
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMDerivedClassAddrOpLowering::matchAndRewrite(
    cir::DerivedClassAddrOp derivedClassOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto resultType =
      getTypeConverter()->convertType(derivedClassOp.getType());
  mlir::Value baseAddr = adaptor.getBaseAddr();
  int64_t offsetVal = adaptor.getOffset().getZExtValue() * -1;
  llvm::SmallVector<mlir::LLVM::GEPArg, 1> offset = {offsetVal};
  mlir::Type byteType = mlir::IntegerType::get(resultType.getContext(), 8,
                                               mlir::IntegerType::Signless);
  if (derivedClassOp.getAssumeNotNull()) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(derivedClassOp, resultType,
                                                   byteType, baseAddr, offset);
  } else {
    auto loc = derivedClassOp.getLoc();
    mlir::Value isNull = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, baseAddr,
        rewriter.create<mlir::LLVM::ZeroOp>(loc, baseAddr.getType()));
    mlir::Value adjusted = rewriter.create<mlir::LLVM::GEPOp>(
        loc, resultType, byteType, baseAddr, offset);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(derivedClassOp, isNull,
                                                      baseAddr, adjusted);
  }
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBaseDataMemberOpLowering::matchAndRewrite(
    cir::BaseDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerMod->getCXXABI().lowerBaseDataMember(op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMDerivedDataMemberOpLowering::matchAndRewrite(
    cir::DerivedDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerMod->getCXXABI().lowerDerivedDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

static mlir::Value
getValueForVTableSymbol(mlir::Operation *op,
                        mlir::ConversionPatternRewriter &rewriter,
                        const mlir::TypeConverter *converter,
                        mlir::FlatSymbolRefAttr nameAttr, mlir::Type &eltType) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  auto *symbol = mlir::SymbolTable::lookupSymbolIn(module, nameAttr);
  if (auto llvmSymbol = dyn_cast<mlir::LLVM::GlobalOp>(symbol)) {
    eltType = llvmSymbol.getType();
  } else if (auto cirSymbol = dyn_cast<cir::GlobalOp>(symbol)) {
    eltType = converter->convertType(cirSymbol.getSymType());
  }
  return rewriter.create<mlir::LLVM::AddressOfOp>(
      op->getLoc(), mlir::LLVM::LLVMPointerType::get(op->getContext()),
      nameAttr.getValue());
}

mlir::LogicalResult CIRToLLVMVTTAddrPointOpLowering::matchAndRewrite(
    cir::VTTAddrPointOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const mlir::Type resultType = getTypeConverter()->convertType(op.getType());
  llvm::SmallVector<mlir::LLVM::GEPArg> offsets;
  mlir::Type eltType;
  mlir::Value llvmAddr = adaptor.getSymAddr();

  if (op.getSymAddr()) {
    if (op.getOffset() == 0) {
      rewriter.replaceAllUsesWith(op, llvmAddr);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    offsets.push_back(adaptor.getOffset());
    eltType = mlir::IntegerType::get(resultType.getContext(), 8,
                                     mlir::IntegerType::Signless);
  } else {
    llvmAddr = getValueForVTableSymbol(op, rewriter, getTypeConverter(),
                                       op.getNameAttr(), eltType);
    assert(eltType && "Shouldn't ever be missing an eltType here");
    offsets.push_back(0);
    offsets.push_back(adaptor.getOffset());
  }
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, resultType, eltType,
                                                 llvmAddr, offsets, true);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBrCondOpLowering::matchAndRewrite(
    cir::BrCondOp brOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value i1Condition;

  auto hasOneUse = false;

  if (auto defOp = brOp.getCond().getDefiningOp())
    hasOneUse = defOp->getResult(0).hasOneUse();

  if (auto defOp = adaptor.getCond().getDefiningOp()) {
    if (auto zext = dyn_cast<mlir::LLVM::ZExtOp>(defOp)) {
      if (zext->use_empty() &&
          zext->getOperand(0).getType() == rewriter.getI1Type()) {
        i1Condition = zext->getOperand(0);
        if (hasOneUse)
          rewriter.eraseOp(zext);
      }
    }
  }

  if (!i1Condition)
    i1Condition = rewriter.create<mlir::LLVM::TruncOp>(
        brOp.getLoc(), rewriter.getI1Type(), adaptor.getCond());

  rewriter.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
      brOp, i1Condition, brOp.getDestTrue(), adaptor.getDestOperandsTrue(),
      brOp.getDestFalse(), adaptor.getDestOperandsFalse());

  return mlir::success();
}

mlir::Type CIRToLLVMCastOpLowering::convertTy(mlir::Type ty) const {
  return getTypeConverter()->convertType(ty);
}

mlir::LogicalResult CIRToLLVMCastOpLowering::matchAndRewrite(
    cir::CastOp castOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // For arithmetic conversions, LLVM IR uses the same instruction to convert
  // both individual scalars and entire vectors. This lowering pass handles
  // both situations.

  auto src = adaptor.getSrc();

  switch (castOp.getKind()) {
  case cir::CastKind::array_to_ptrdecay: {
    const auto ptrTy = mlir::cast<cir::PointerType>(castOp.getType());
    auto sourceValue = adaptor.getOperands().front();
    auto targetType = convertTy(ptrTy);
    auto elementTy = convertTy(ptrTy.getPointee());
    auto offset = llvm::SmallVector<mlir::LLVM::GEPArg>{0};
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        castOp, targetType, elementTy, sourceValue, offset);
    break;
  }
  case cir::CastKind::int_to_bool: {
    auto zero = rewriter.create<cir::ConstantOp>(
        src.getLoc(), castOp.getSrc().getType(),
        cir::IntAttr::get(castOp.getSrc().getType(), 0));
    rewriter.replaceOpWithNewOp<cir::CmpOp>(
        castOp, cir::BoolType::get(getContext()), cir::CmpOpKind::ne,
        castOp.getSrc(), zero);
    break;
  }
  case cir::CastKind::integral: {
    auto srcType = castOp.getSrc().getType();
    auto dstType = castOp.getResult().getType();
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstType = getTypeConverter()->convertType(dstType);
    cir::IntType srcIntType =
        mlir::cast<cir::IntType>(elementTypeIfVector(srcType));
    cir::IntType dstIntType =
        mlir::cast<cir::IntType>(elementTypeIfVector(dstType));
    rewriter.replaceOp(castOp, getLLVMIntCast(rewriter, llvmSrcVal, llvmDstType,
                                              srcIntType.isUnsigned(),
                                              srcIntType.getWidth(),
                                              dstIntType.getWidth()));
    break;
  }
  case cir::CastKind::floating: {
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy =
        getTypeConverter()->convertType(castOp.getResult().getType());

    auto srcTy = elementTypeIfVector(castOp.getSrc().getType());
    auto dstTy = elementTypeIfVector(castOp.getResult().getType());

    if (!mlir::isa<cir::CIRFPTypeInterface>(dstTy) ||
        !mlir::isa<cir::CIRFPTypeInterface>(srcTy))
      return castOp.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

    auto getFloatWidth = [](mlir::Type ty) -> unsigned {
      return mlir::cast<cir::CIRFPTypeInterface>(ty).getWidth();
    };

    if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPExtOp>(castOp, llvmDstTy,
                                                       llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::int_to_ptr: {
    auto dstTy = mlir::cast<cir::PointerType>(castOp.getType());
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::ptr_to_int: {
    auto dstTy = mlir::cast<cir::IntType>(castOp.getType());
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::float_to_bool: {
    auto dstTy = mlir::cast<cir::BoolType>(castOp.getType());
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    auto kind = mlir::LLVM::FCmpPredicate::une;

    // Check if float is not equal to zero.
    auto zeroFloat = rewriter.create<mlir::LLVM::ConstantOp>(
        castOp.getLoc(), llvmSrcVal.getType(),
        mlir::FloatAttr::get(llvmSrcVal.getType(), 0.0));

    // Extend comparison result to either bool (C++) or int (C).
    mlir::Value cmpResult = rewriter.create<mlir::LLVM::FCmpOp>(
        castOp.getLoc(), kind, llvmSrcVal, zeroFloat);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                    cmpResult);
    return mlir::success();
  }
  case cir::CastKind::bool_to_int: {
    auto dstTy = mlir::cast<cir::IntType>(castOp.getType());
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmSrcTy = mlir::cast<mlir::IntegerType>(llvmSrcVal.getType());
    auto llvmDstTy =
        mlir::cast<mlir::IntegerType>(getTypeConverter()->convertType(dstTy));
    if (llvmSrcTy.getWidth() == llvmDstTy.getWidth())
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::bool_to_float: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                      llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::int_to_float: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (mlir::cast<cir::IntType>(elementTypeIfVector(castOp.getSrc().getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::float_to_int: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    if (mlir::cast<cir::IntType>(
            elementTypeIfVector(castOp.getResult().getType()))
            .isSigned())
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FPToUIOp>(castOp, llvmDstTy,
                                                        llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::bitcast: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(castOp, llvmDstTy,
                                                       llvmSrcVal);
    return mlir::success();
  }
  case cir::CastKind::ptr_to_bool: {
    auto zero =
        mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 64), 0);
    auto null = rewriter.create<cir::ConstantOp>(
        src.getLoc(), castOp.getSrc().getType(),
        cir::ConstPtrAttr::get(getContext(), castOp.getSrc().getType(), zero));
    rewriter.replaceOpWithNewOp<cir::CmpOp>(
        castOp, cir::BoolType::get(getContext()), cir::CmpOpKind::ne,
        castOp.getSrc(), null);
    break;
  }
  case cir::CastKind::address_space: {
    auto dstTy = castOp.getType();
    auto llvmSrcVal = adaptor.getOperands().front();
    auto llvmDstTy = getTypeConverter()->convertType(dstTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddrSpaceCastOp>(castOp, llvmDstTy,
                                                             llvmSrcVal);
    break;
  }
  default: {
    return castOp.emitError("Unhandled cast kind: ")
           << castOp.getKindAttrName();
  }
  }

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMReturnOpLowering::matchAndRewrite(
    cir::ReturnOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
  return mlir::LogicalResult::success();
}

struct ConvertCIRToLLVMPass
    : public mlir::PassWrapper<ConvertCIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::DLTIDialect,
                    mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();
  }
  void runOnOperation() final;

  void buildGlobalAnnotationsVar(
      llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
      llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
      llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap);

  void processCIRAttrs(mlir::ModuleOp moduleOp);

  virtual StringRef getArgument() const override { return "cir-flat-to-llvm"; }
};

mlir::LogicalResult
rewriteToCallOrInvoke(mlir::Operation *op, mlir::ValueRange callOperands,
                      mlir::ConversionPatternRewriter &rewriter,
                      const mlir::TypeConverter *converter,
                      mlir::FlatSymbolRefAttr calleeAttr,
                      mlir::Block *continueBlock = nullptr,
                      mlir::Block *landingPadBlock = nullptr) {
  llvm::SmallVector<mlir::Type, 8> llvmResults;
  auto cirResults = op->getResultTypes();
  auto callIf = cast<cir::CIRCallOpInterface>(op);

  if (converter->convertTypes(cirResults, llvmResults).failed())
    return mlir::failure();

  auto cconv = convertCallingConv(callIf.getCallingConv());

  mlir::LLVM::LLVMFunctionType llvmFnTy;
  if (calleeAttr) { // direct call
    auto fn =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::FunctionOpInterface>(
            op, calleeAttr);
    assert(fn && "Did not find function for call");
    llvmFnTy = cast<mlir::LLVM::LLVMFunctionType>(
        converter->convertType(fn.getFunctionType()));
  } else { // indirect call
    assert(op->getOperands().size() &&
           "operands list must no be empty for the indirect call");
    auto typ = op->getOperands().front().getType();
    assert(isa<cir::PointerType>(typ) && "expected pointer type");
    auto ptyp = dyn_cast<cir::PointerType>(typ);
    auto ftyp = dyn_cast<cir::FuncType>(ptyp.getPointee());
    assert(ftyp && "expected a pointer to a function as the first operand");
    llvmFnTy = cast<mlir::LLVM::LLVMFunctionType>(converter->convertType(ftyp));
  }

  if (landingPadBlock) {
    auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
        op, llvmFnTy, calleeAttr, callOperands, continueBlock,
        mlir::ValueRange{}, landingPadBlock, mlir::ValueRange{});
    newOp.setCConv(cconv);
  } else {
    auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, llvmFnTy, calleeAttr, callOperands);
    newOp.setCConv(cconv);
  }
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCallOpLowering::matchAndRewrite(
    cir::CallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  return rewriteToCallOrInvoke(op.getOperation(), adaptor.getOperands(),
                               rewriter, getTypeConverter(),
                               op.getCalleeAttr());
}

mlir::LogicalResult CIRToLLVMTryCallOpLowering::matchAndRewrite(
    cir::TryCallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (op.getCallingConv() != cir::CallingConv::C) {
    return op.emitError(
        "non-C calling convention is not implemented for try_call");
  }
  return rewriteToCallOrInvoke(op.getOperation(), adaptor.getOperands(),
                               rewriter, getTypeConverter(), op.getCalleeAttr(),
                               op.getCont(), op.getLandingPad());
}

static mlir::LLVM::LLVMStructType
getLLVMLandingPadStructTy(mlir::ConversionPatternRewriter &rewriter) {
  // Create the landing pad type: struct { ptr, i32 }
  mlir::MLIRContext *ctx = rewriter.getContext();
  auto llvmPtr = mlir::LLVM::LLVMPointerType::get(ctx);
  llvm::SmallVector<mlir::Type> structFields;
  structFields.push_back(llvmPtr);
  structFields.push_back(rewriter.getI32Type());

  return mlir::LLVM::LLVMStructType::getLiteral(ctx, structFields);
}

mlir::LogicalResult CIRToLLVMEhInflightOpLowering::matchAndRewrite(
    cir::EhInflightOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  auto llvmLandingPadStructTy = getLLVMLandingPadStructTy(rewriter);
  mlir::ArrayAttr symListAttr = op.getSymTypeListAttr();
  mlir::SmallVector<mlir::Value, 4> symAddrs;

  auto llvmFn = op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  assert(llvmFn && "expected LLVM function parent");
  mlir::Block *entryBlock = &llvmFn.getRegion().front();
  assert(entryBlock->isEntryBlock());

  // %x = landingpad { ptr, i32 }
  // Note that since llvm.landingpad has to be the first operation on the
  // block, any needed value for its operands has to be added somewhere else.
  if (symListAttr) {
    //   catch ptr @_ZTIi
    //   catch ptr @_ZTIPKc
    for (mlir::Attribute attr : op.getSymTypeListAttr()) {
      auto symAttr = cast<mlir::FlatSymbolRefAttr>(attr);
      // Generate `llvm.mlir.addressof` for each symbol, and place those
      // operations in the LLVM function entry basic block.
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);
      mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
          loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
          symAttr.getValue());
      symAddrs.push_back(addrOp);
    }
  } else {
    if (!op.getCleanup()) {
      //   catch ptr null
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entryBlock);
      mlir::Value nullOp = rewriter.create<mlir::LLVM::ZeroOp>(
          loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
      symAddrs.push_back(nullOp);
    }
  }

  // %slot = extractvalue { ptr, i32 } %x, 0
  // %selector = extractvalue { ptr, i32 } %x, 1
  auto padOp = rewriter.create<mlir::LLVM::LandingpadOp>(
      loc, llvmLandingPadStructTy, symAddrs);
  SmallVector<int64_t> slotIdx = {0};
  SmallVector<int64_t> selectorIdx = {1};

  if (op.getCleanup())
    padOp.setCleanup(true);

  mlir::Value slot =
      rewriter.create<mlir::LLVM::ExtractValueOp>(loc, padOp, slotIdx);
  mlir::Value selector =
      rewriter.create<mlir::LLVM::ExtractValueOp>(loc, padOp, selectorIdx);

  rewriter.replaceOp(op, mlir::ValueRange{slot, selector});

  // Landing pads are required to be in LLVM functions with personality
  // attribute. FIXME: for now hardcode personality creation in order to start
  // adding exception tests, once we annotate CIR with such information,
  // change it to be in FuncOp lowering instead.
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    // Insert personality decl before the current function.
    rewriter.setInsertionPoint(llvmFn);
    auto personalityFnTy =
        mlir::LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {},
                                          /*isVarArg=*/true);
    // Get or create `__gxx_personality_v0`
    StringRef fnName = "__gxx_personality_v0";
    getOrCreateLLVMFuncOp(rewriter, op, fnName, personalityFnTy);
    llvmFn.setPersonality(fnName);
  }
  return mlir::success();
}

void CIRToLLVMAllocaOpLowering::buildAllocaAnnotations(
    mlir::LLVM::AllocaOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter,
    mlir::ArrayAttr annotationValuesArray) const {
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  mlir::OpBuilder globalVarBuilder(module.getContext());

  mlir::OpBuilder::InsertPoint afterAlloca = rewriter.saveInsertionPoint();
  globalVarBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());

  mlir::Location loc = op.getLoc();
  mlir::OpBuilder varInitBuilder(module.getContext());
  varInitBuilder.restoreInsertionPoint(afterAlloca);

  auto intrinRetTy = mlir::LLVM::LLVMVoidType::get(getContext());
  constexpr const char *intrinNameAttr = "llvm.var.annotation.p0.p0";
  for (mlir::Attribute entry : annotationValuesArray) {
    SmallVector<mlir::Value, 4> intrinsicArgs;
    intrinsicArgs.push_back(op.getRes());
    auto annot = cast<cir::AnnotationAttr>(entry);
    lowerAnnotationValue(loc, loc, annot, module, varInitBuilder,
                         globalVarBuilder, stringGlobalsMap,
                         argStringGlobalsMap, argsVarMap, intrinsicArgs);
    rewriter.create<mlir::LLVM::CallIntrinsicOp>(
        loc, intrinRetTy, mlir::StringAttr::get(getContext(), intrinNameAttr),
        intrinsicArgs);
  }
}

mlir::LogicalResult CIRToLLVMAllocaOpLowering::matchAndRewrite(
    cir::AllocaOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value size =
      op.isDynamic() ? adaptor.getDynAllocSize()
                     : rewriter.create<mlir::LLVM::ConstantOp>(
                           op.getLoc(),
                           typeConverter->convertType(rewriter.getIndexType()),
                           rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
  auto elementTy = getTypeConverter()->convertType(op.getAllocaType());
  auto resultTy = getTypeConverter()->convertType(op.getResult().getType());
  // Verification between the CIR alloca AS and the one from data layout.
  {
    auto resPtrTy = mlir::cast<mlir::LLVM::LLVMPointerType>(resultTy);
    auto dlAllocaASAttr = mlir::cast_if_present<mlir::IntegerAttr>(
        dataLayout.getAllocaMemorySpace());
    // Absence means 0
    // TODO: The query for the alloca AS should be done through CIRDataLayout
    // instead to reuse the logic of interpret null attr as 0.
    auto dlAllocaAS = dlAllocaASAttr ? dlAllocaASAttr.getInt() : 0;
    if (dlAllocaAS != resPtrTy.getAddressSpace()) {
      return op.emitError() << "alloca address space doesn't match the one "
                               "from the target data layout: "
                            << dlAllocaAS;
    }
  }

  // If there are annotations available, copy them out before we destroy the
  // original cir.alloca.
  mlir::ArrayAttr annotations;
  if (op.getAnnotations())
    annotations = op.getAnnotationsAttr();

  auto llvmAlloca = rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
      op, resultTy, elementTy, size, op.getAlignmentAttr().getInt());

  if (annotations && !annotations.empty())
    buildAllocaAnnotations(llvmAlloca, adaptor, rewriter, annotations);
  return mlir::success();
}

mlir::LLVM::AtomicOrdering
getLLVMMemOrder(std::optional<cir::MemOrder> &memorder) {
  if (!memorder)
    return mlir::LLVM::AtomicOrdering::not_atomic;
  switch (*memorder) {
  case cir::MemOrder::Relaxed:
    return mlir::LLVM::AtomicOrdering::monotonic;
  case cir::MemOrder::Consume:
  case cir::MemOrder::Acquire:
    return mlir::LLVM::AtomicOrdering::acquire;
  case cir::MemOrder::Release:
    return mlir::LLVM::AtomicOrdering::release;
  case cir::MemOrder::AcquireRelease:
    return mlir::LLVM::AtomicOrdering::acq_rel;
  case cir::MemOrder::SequentiallyConsistent:
    return mlir::LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("unknown memory order");
}

mlir::LogicalResult CIRToLLVMLoadOpLowering::matchAndRewrite(
    cir::LoadOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto llvmTy = getTypeConverter()->convertType(op.getResult().getType());
  auto memorder = op.getMemOrder();
  auto ordering = getLLVMMemOrder(memorder);
  auto alignOpt = op.getAlignment();
  unsigned alignment = 0;
  if (!alignOpt) {
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    alignment = (unsigned)layout.getTypeABIAlignment(llvmTy);
  } else {
    alignment = *alignOpt;
  }

  // TODO: nontemporal, invariant, syncscope.
  auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(
      op->getLoc(), llvmTy, adaptor.getAddr(), /* alignment */ alignment,
      op.getIsVolatile(), /* nontemporal */ false,
      /* invariant */ false, /* invariantGroup */ false, ordering);
  rewriter.replaceOp(op, loadOp);
  loadOp.setTBAATags(op.getTbaaAttr());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMStoreOpLowering::matchAndRewrite(
    cir::StoreOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto memorder = op.getMemOrder();
  auto ordering = getLLVMMemOrder(memorder);
  auto alignOpt = op.getAlignment();
  unsigned alignment = 0;

  if (!alignOpt) {
    const auto llvmTy =
        getTypeConverter()->convertType(op.getValue().getType());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    alignment = (unsigned)layout.getTypeABIAlignment(llvmTy);
  } else {
    alignment = *alignOpt;
  }

  // TODO: nontemporal, syncscope.
  auto storeOp = rewriter.create<mlir::LLVM::StoreOp>(
      op->getLoc(), adaptor.getValue(), adaptor.getAddr(), alignment,
      op.getIsVolatile(),
      /* nontemporal */ false, /* invariantGroup */ false, ordering);
  rewriter.replaceOp(op, storeOp);
  storeOp.setTBAATags(op.getTbaaAttr());
  return mlir::LogicalResult::success();
}

bool hasTrailingZeros(cir::ConstArrayAttr attr) {
  auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr.getElts());
  return attr.hasTrailingZeros() ||
         (array && std::count_if(array.begin(), array.end(), [](auto elt) {
            auto ar = dyn_cast<cir::ConstArrayAttr>(elt);
            return ar && hasTrailingZeros(ar);
          }));
}

mlir::LogicalResult CIRToLLVMConstantOpLowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Attribute attr = op.getValue();

  if (mlir::isa<mlir::IntegerType>(op.getType())) {
    // Verified cir.const operations cannot actually be of these types, but the
    // lowering pass may generate temporary cir.const operations with these
    // types. This is OK since MLIR allows unverified operations to be alive
    // during a pass as long as they don't live past the end of the pass.
    attr = op.getValue();
  } else if (mlir::isa<cir::BoolType>(op.getType())) {
    int value = (op.getValue() ==
                 cir::BoolAttr::get(getContext(),
                                    cir::BoolType::get(getContext()), true));
    attr = rewriter.getIntegerAttr(typeConverter->convertType(op.getType()),
                                   value);
  } else if (mlir::isa<cir::IntType>(op.getType())) {
    attr = rewriter.getIntegerAttr(
        typeConverter->convertType(op.getType()),
        mlir::cast<cir::IntAttr>(op.getValue()).getValue());
  } else if (mlir::isa<cir::CIRFPTypeInterface>(op.getType())) {
    attr = rewriter.getFloatAttr(
        typeConverter->convertType(op.getType()),
        mlir::cast<cir::FPAttr>(op.getValue()).getValue());
  } else if (auto complexTy = mlir::dyn_cast<cir::ComplexType>(op.getType())) {
    auto complexAttr = mlir::cast<cir::ComplexAttr>(op.getValue());
    auto complexElemTy = complexTy.getElementTy();
    auto complexElemLLVMTy = typeConverter->convertType(complexElemTy);

    mlir::Attribute components[2];
    if (mlir::isa<cir::IntType>(complexElemTy)) {
      components[0] = rewriter.getIntegerAttr(
          complexElemLLVMTy,
          mlir::cast<cir::IntAttr>(complexAttr.getReal()).getValue());
      components[1] = rewriter.getIntegerAttr(
          complexElemLLVMTy,
          mlir::cast<cir::IntAttr>(complexAttr.getImag()).getValue());
    } else {
      components[0] = rewriter.getFloatAttr(
          complexElemLLVMTy,
          mlir::cast<cir::FPAttr>(complexAttr.getReal()).getValue());
      components[1] = rewriter.getFloatAttr(
          complexElemLLVMTy,
          mlir::cast<cir::FPAttr>(complexAttr.getImag()).getValue());
    }

    attr = rewriter.getArrayAttr(components);
  } else if (mlir::isa<cir::PointerType>(op.getType())) {
    // Optimize with dedicated LLVM op for null pointers.
    if (mlir::isa<cir::ConstPtrAttr>(op.getValue())) {
      if (mlir::cast<cir::ConstPtrAttr>(op.getValue()).isNullValue()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
            op, typeConverter->convertType(op.getType()));
        return mlir::success();
      }
    }
    // Lower GlobalViewAttr to llvm.mlir.addressof
    if (auto gv = mlir::dyn_cast<cir::GlobalViewAttr>(op.getValue())) {
      auto newOp = lowerCirAttrAsValue(op, gv, rewriter, getTypeConverter());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    attr = op.getValue();
  } else if (mlir::isa<cir::DataMemberType>(op.getType())) {
    assert(lowerMod && "lower module is not available");
    auto dataMember = mlir::cast<cir::DataMemberAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerMod->getCXXABI().lowerDataMemberConstant(
        dataMember, layout, *typeConverter);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }
  // TODO(cir): constant arrays are currently just pushed into the stack using
  // the store instruction, instead of being stored as global variables and
  // then memcopyied into the stack (as done in Clang).
  else if (auto arrTy = mlir::dyn_cast<cir::ArrayType>(op.getType())) {
    // Fetch operation constant array initializer.

    auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(op.getValue());
    if (!constArr && !isa<cir::ZeroAttr, cir::UndefAttr>(op.getValue()))
      return op.emitError() << "array does not have a constant initializer";

    std::optional<mlir::Attribute> denseAttr;
    if (constArr && hasTrailingZeros(constArr)) {
      auto newOp =
          lowerCirAttrAsValue(op, constArr, rewriter, getTypeConverter());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    } else if (constArr &&
               (denseAttr = lowerConstArrayAttr(constArr, typeConverter))) {
      attr = denseAttr.value();
    } else {
      auto initVal =
          lowerCirAttrAsValue(op, op.getValue(), rewriter, typeConverter);
      rewriter.replaceAllUsesWith(op, initVal);
      rewriter.eraseOp(op);
      return mlir::success();
    }
  } else if (const auto structAttr =
                 mlir::dyn_cast<cir::ConstStructAttr>(op.getValue())) {
    // TODO(cir): this diverges from traditional lowering. Normally the
    // initializer would be a global constant that is memcopied. Here we just
    // define a local constant with llvm.undef that will be stored into the
    // stack.
    auto initVal = lowerCirAttrAsValue(op, structAttr, rewriter, typeConverter);
    rewriter.replaceAllUsesWith(op, initVal);
    rewriter.eraseOp(op);
    return mlir::success();
  } else if (auto strTy = mlir::dyn_cast<cir::StructType>(op.getType())) {
    auto attr = op.getValue();
    if (mlir::isa<cir::ZeroAttr, cir::UndefAttr>(attr)) {
      auto initVal = lowerCirAttrAsValue(op, attr, rewriter, typeConverter);
      rewriter.replaceAllUsesWith(op, initVal);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return op.emitError() << "unsupported lowering for struct constant type "
                          << op.getType();
  } else if (const auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType())) {
    rewriter.replaceOp(op, lowerCirAttrAsValue(op, op.getValue(), rewriter,
                                               getTypeConverter()));
    return mlir::success();
  } else
    return op.emitError() << "unsupported constant type " << op.getType();

  rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
      op, getTypeConverter()->convertType(op.getType()), attr);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecCreateOpLowering::matchAndRewrite(
    cir::VecCreateOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Start with an 'undef' value for the vector.  Then 'insertelement' for
  // each of the vector elements.
  auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType());
  assert(vecTy && "result type of cir.vec.create op is not VectorType");
  auto llvmTy = typeConverter->convertType(vecTy);
  auto loc = op.getLoc();
  mlir::Value result = rewriter.create<mlir::LLVM::PoisonOp>(loc, llvmTy);
  assert(vecTy.getSize() == op.getElements().size() &&
         "cir.vec.create op count doesn't match vector type elements count");
  for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
    mlir::Value indexValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), i);
    result = rewriter.create<mlir::LLVM::InsertElementOp>(
        loc, result, adaptor.getElements()[i], indexValue);
  }
  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecCmpOpLowering::matchAndRewrite(
    cir::VecCmpOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(mlir::isa<cir::VectorType>(op.getType()) &&
         mlir::isa<cir::VectorType>(op.getLhs().getType()) &&
         mlir::isa<cir::VectorType>(op.getRhs().getType()) &&
         "Vector compare with non-vector type");
  // LLVM IR vector comparison returns a vector of i1.  This one-bit vector
  // must be sign-extended to the correct result type.
  auto elementType = elementTypeIfVector(op.getLhs().getType());
  mlir::Value bitResult;
  if (auto intType = mlir::dyn_cast<cir::IntType>(elementType)) {
    bitResult = rewriter.create<mlir::LLVM::ICmpOp>(
        op.getLoc(),
        convertCmpKindToICmpPredicate(op.getKind(), intType.isSigned()),
        adaptor.getLhs(), adaptor.getRhs());
  } else if (mlir::isa<cir::CIRFPTypeInterface>(elementType)) {
    bitResult = rewriter.create<mlir::LLVM::FCmpOp>(
        op.getLoc(), convertCmpKindToFCmpPredicate(op.getKind()),
        adaptor.getLhs(), adaptor.getRhs());
  } else {
    return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
  }
  rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(
      op, typeConverter->convertType(op.getType()), bitResult);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecSplatOpLowering::matchAndRewrite(
    cir::VecSplatOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Vector splat can be implemented with an `insertelement` and a
  // `shufflevector`, which is better than an `insertelement` for each
  // element in the vector. Start with an undef vector. Insert the value into
  // the first element. Then use a `shufflevector` with a mask of all 0 to
  // fill out the entire vector with that value.
  auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType());
  assert(vecTy && "result type of cir.vec.splat op is not VectorType");
  auto llvmTy = typeConverter->convertType(vecTy);
  auto loc = op.getLoc();
  mlir::Value undef = rewriter.create<mlir::LLVM::PoisonOp>(loc, llvmTy);
  mlir::Value indexValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
  mlir::Value elementValue = adaptor.getValue();
  mlir::Value oneElement = rewriter.create<mlir::LLVM::InsertElementOp>(
      loc, undef, elementValue, indexValue);
  SmallVector<int32_t> zeroValues(vecTy.getSize(), 0);
  mlir::Value shuffled = rewriter.create<mlir::LLVM::ShuffleVectorOp>(
      loc, oneElement, undef, zeroValues);
  rewriter.replaceOp(op, shuffled);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecTernaryOpLowering::matchAndRewrite(
    cir::VecTernaryOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(mlir::isa<cir::VectorType>(op.getType()) &&
         mlir::isa<cir::VectorType>(op.getCond().getType()) &&
         mlir::isa<cir::VectorType>(op.getVec1().getType()) &&
         mlir::isa<cir::VectorType>(op.getVec2().getType()) &&
         "Vector ternary op with non-vector type");
  // Convert `cond` into a vector of i1, then use that in a `select` op.
  mlir::Value bitVec = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(), mlir::LLVM::ICmpPredicate::ne, adaptor.getCond(),
      rewriter.create<mlir::LLVM::ZeroOp>(
          op.getCond().getLoc(),
          typeConverter->convertType(op.getCond().getType())));
  rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
      op, bitVec, adaptor.getVec1(), adaptor.getVec2());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecShuffleOpLowering::matchAndRewrite(
    cir::VecShuffleOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // LLVM::ShuffleVectorOp takes an ArrayRef of int for the list of indices.
  // Convert the ClangIR ArrayAttr of IntAttr constants into a
  // SmallVector<int>.
  SmallVector<int, 8> indices;
  std::transform(
      op.getIndices().begin(), op.getIndices().end(),
      std::back_inserter(indices), [](mlir::Attribute intAttr) {
        return mlir::cast<cir::IntAttr>(intAttr).getValue().getSExtValue();
      });
  rewriter.replaceOpWithNewOp<mlir::LLVM::ShuffleVectorOp>(
      op, adaptor.getVec1(), adaptor.getVec2(), indices);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVecShuffleDynamicOpLowering::matchAndRewrite(
    cir::VecShuffleDynamicOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // LLVM IR does not have an operation that corresponds to this form of
  // the built-in.
  //     __builtin_shufflevector(V, I)
  // is implemented as this pseudocode, where the for loop is unrolled
  // and N is the number of elements:
  //     masked = I & (N-1)
  //     for (i in 0 <= i < N)
  //       result[i] = V[masked[i]]
  auto loc = op.getLoc();
  mlir::Value input = adaptor.getVec();
  mlir::Type llvmIndexVecType =
      getTypeConverter()->convertType(op.getIndices().getType());
  mlir::Type llvmIndexType = getTypeConverter()->convertType(
      elementTypeIfVector(op.getIndices().getType()));
  uint64_t numElements =
      mlir::cast<cir::VectorType>(op.getVec().getType()).getSize();
  mlir::Value maskValue = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, llvmIndexType,
      mlir::IntegerAttr::get(llvmIndexType, numElements - 1));
  mlir::Value maskVector =
      rewriter.create<mlir::LLVM::UndefOp>(loc, llvmIndexVecType);
  for (uint64_t i = 0; i < numElements; ++i) {
    mlir::Value iValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), i);
    maskVector = rewriter.create<mlir::LLVM::InsertElementOp>(
        loc, maskVector, maskValue, iValue);
  }
  mlir::Value maskedIndices = rewriter.create<mlir::LLVM::AndOp>(
      loc, llvmIndexVecType, adaptor.getIndices(), maskVector);
  mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(
      loc, getTypeConverter()->convertType(op.getVec().getType()));
  for (uint64_t i = 0; i < numElements; ++i) {
    mlir::Value iValue =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI64Type(), i);
    mlir::Value indexValue = rewriter.create<mlir::LLVM::ExtractElementOp>(
        loc, maskedIndices, iValue);
    mlir::Value valueAtIndex =
        rewriter.create<mlir::LLVM::ExtractElementOp>(loc, input, indexValue);
    result = rewriter.create<mlir::LLVM::InsertElementOp>(loc, result,
                                                          valueAtIndex, iValue);
  }
  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVAStartOpLowering::matchAndRewrite(
    cir::VAStartOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
  auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(
      op.getLoc(), opaquePtr, adaptor.getOperands().front());
  rewriter.replaceOpWithNewOp<mlir::LLVM::VaStartOp>(op, vaList);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVAEndOpLowering::matchAndRewrite(
    cir::VAEndOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
  auto vaList = rewriter.create<mlir::LLVM::BitcastOp>(
      op.getLoc(), opaquePtr, adaptor.getOperands().front());
  rewriter.replaceOpWithNewOp<mlir::LLVM::VaEndOp>(op, vaList);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVACopyOpLowering::matchAndRewrite(
    cir::VACopyOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto opaquePtr = mlir::LLVM::LLVMPointerType::get(getContext());
  auto dstList = rewriter.create<mlir::LLVM::BitcastOp>(
      op.getLoc(), opaquePtr, adaptor.getOperands().front());
  auto srcList = rewriter.create<mlir::LLVM::BitcastOp>(
      op.getLoc(), opaquePtr, adaptor.getOperands().back());
  rewriter.replaceOpWithNewOp<mlir::LLVM::VaCopyOp>(op, dstList, srcList);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVAArgOpLowering::matchAndRewrite(
    cir::VAArgOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  return op.emitError("cir.vaarg lowering is NYI");
}

/// Returns the name used for the linkage attribute. This *must* correspond
/// to the name of the attribute in ODS.
StringRef CIRToLLVMFuncOpLowering::getLinkageAttrNameString() {
  return "linkage";
}

/// Convert the `cir.func` attributes to `llvm.func` attributes.
/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out
/// argument attributes.
void CIRToLLVMFuncOpLowering::lowerFuncAttributes(
    cir::FuncOp func, bool filterArgAndResAttrs,
    SmallVectorImpl<mlir::NamedAttribute> &result) const {
  for (auto attr : func->getAttrs()) {
    if (attr.getName() == mlir::SymbolTable::getSymbolAttrName() ||
        attr.getName() == func.getFunctionTypeAttrName() ||
        attr.getName() == getLinkageAttrNameString() ||
        attr.getName() == func.getCallingConvAttrName() ||
        (filterArgAndResAttrs &&
         (attr.getName() == func.getArgAttrsAttrName() ||
          attr.getName() == func.getResAttrsAttrName())))
      continue;

    // `CIRDialectLLVMIRTranslationInterface` requires "cir." prefix for
    // dialect specific attributes, rename them.
    if (attr.getName() == func.getExtraAttrsAttrName()) {
      std::string cirName = "cir." + func.getExtraAttrsAttrName().str();
      attr.setName(mlir::StringAttr::get(getContext(), cirName));

      lowerFuncOpenCLKernelMetadata(attr);
    }
    result.push_back(attr);
  }
}

/// When do module translation, we can only translate LLVM-compatible types.
/// Here we lower possible OpenCLKernelMetadataAttr to use the converted type.
void CIRToLLVMFuncOpLowering::lowerFuncOpenCLKernelMetadata(
    mlir::NamedAttribute &extraAttrsEntry) const {
  const auto attrKey = cir::OpenCLKernelMetadataAttr::getMnemonic();
  auto oldExtraAttrs =
      cast<cir::ExtraFuncAttributesAttr>(extraAttrsEntry.getValue());
  if (!oldExtraAttrs.getElements().contains(attrKey))
    return;

  mlir::NamedAttrList newExtraAttrs;
  for (auto entry : oldExtraAttrs.getElements()) {
    if (entry.getName() == attrKey) {
      auto clKernelMetadata =
          cast<cir::OpenCLKernelMetadataAttr>(entry.getValue());
      if (auto vecTypeHint = clKernelMetadata.getVecTypeHint()) {
        auto newType = typeConverter->convertType(vecTypeHint.getValue());
        auto newTypeHint = mlir::TypeAttr::get(newType);
        auto newCLKMAttr = cir::OpenCLKernelMetadataAttr::get(
            getContext(), clKernelMetadata.getWorkGroupSizeHint(),
            clKernelMetadata.getReqdWorkGroupSize(), newTypeHint,
            clKernelMetadata.getVecTypeHintSignedness(),
            clKernelMetadata.getIntelReqdSubGroupSize());
        entry.setValue(newCLKMAttr);
      }
    }
    newExtraAttrs.push_back(entry);
  }
  extraAttrsEntry.setValue(cir::ExtraFuncAttributesAttr::get(
      getContext(), newExtraAttrs.getDictionary(getContext())));
}

mlir::LogicalResult CIRToLLVMFuncOpLowering::matchAndRewrite(
    cir::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  auto fnType = op.getFunctionType();
  auto isDsoLocal = op.getDsolocal();
  mlir::TypeConverter::SignatureConversion signatureConversion(
      fnType.getNumInputs());

  for (const auto &argType : enumerate(fnType.getInputs())) {
    auto convertedType = typeConverter->convertType(argType.value());
    if (!convertedType)
      return mlir::failure();
    signatureConversion.addInputs(argType.index(), convertedType);
  }

  mlir::Type resultType =
      getTypeConverter()->convertType(fnType.getReturnType());

  // Create the LLVM function operation.
  auto llvmFnTy = mlir::LLVM::LLVMFunctionType::get(
      resultType ? resultType : mlir::LLVM::LLVMVoidType::get(getContext()),
      signatureConversion.getConvertedTypes(),
      /*isVarArg=*/fnType.isVarArg());
  // LLVMFuncOp expects a single FileLine Location instead of a fused
  // location.
  auto Loc = op.getLoc();
  if (mlir::isa<mlir::FusedLoc>(Loc)) {
    auto FusedLoc = mlir::cast<mlir::FusedLoc>(Loc);
    Loc = FusedLoc.getLocations()[0];
  }
  assert((mlir::isa<mlir::FileLineColLoc>(Loc) ||
          mlir::isa<mlir::UnknownLoc>(Loc)) &&
         "expected single location or unknown location here");

  auto linkage = convertLinkage(op.getLinkage());
  auto cconv = convertCallingConv(op.getCallingConv());
  SmallVector<mlir::NamedAttribute, 4> attributes;
  lowerFuncAttributes(op, /*filterArgAndResAttrs=*/false, attributes);

  auto fn = rewriter.create<mlir::LLVM::LLVMFuncOp>(
      Loc, op.getName(), llvmFnTy, linkage, isDsoLocal, cconv,
      mlir::SymbolRefAttr(), attributes);

  fn.setVisibility_Attr(mlir::LLVM::VisibilityAttr::get(
      getContext(), lowerCIRVisibilityToLLVMVisibility(
                        op.getGlobalVisibilityAttr().getValue())));

  rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
  if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                         &signatureConversion)))
    return mlir::failure();

  rewriter.eraseOp(op);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMGetGlobalOpLowering::matchAndRewrite(
    cir::GetGlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
  // CIRGen should mitigate this and not emit the get_global.
  if (op->getUses().empty()) {
    rewriter.eraseOp(op);
    return mlir::success();
  }

  auto type = getTypeConverter()->convertType(op.getType());
  auto symbol = op.getName();
  mlir::Operation *newop =
      rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), type, symbol);

  if (op.getTls()) {
    // Handle access to TLS via intrinsic.
    newop = rewriter.create<mlir::LLVM::ThreadlocalAddressOp>(
        op.getLoc(), type, newop->getResult(0));
  }

  rewriter.replaceOp(op, newop);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexCreateOpLowering::matchAndRewrite(
    cir::ComplexCreateOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto complexLLVMTy =
      getTypeConverter()->convertType(op.getResult().getType());
  auto initialComplex =
      rewriter.create<mlir::LLVM::UndefOp>(op->getLoc(), complexLLVMTy);

  int64_t position[1]{0};
  auto realComplex = rewriter.create<mlir::LLVM::InsertValueOp>(
      op->getLoc(), initialComplex, adaptor.getReal(), position);

  position[0] = 1;
  auto complex = rewriter.create<mlir::LLVM::InsertValueOp>(
      op->getLoc(), realComplex, adaptor.getImag(), position);

  rewriter.replaceOp(op, complex);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexRealOpLowering::matchAndRewrite(
    cir::ComplexRealOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resultLLVMTy = getTypeConverter()->convertType(op.getResult().getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
      op, resultLLVMTy, adaptor.getOperand(), llvm::ArrayRef<std::int64_t>{0});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexImagOpLowering::matchAndRewrite(
    cir::ComplexImagOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resultLLVMTy = getTypeConverter()->convertType(op.getResult().getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
      op, resultLLVMTy, adaptor.getOperand(), llvm::ArrayRef<std::int64_t>{1});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexRealPtrOpLowering::matchAndRewrite(
    cir::ComplexRealPtrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto operandTy = mlir::cast<cir::PointerType>(op.getOperand().getType());
  auto resultLLVMTy = getTypeConverter()->convertType(op.getResult().getType());
  auto elementLLVMTy = getTypeConverter()->convertType(operandTy.getPointee());

  mlir::LLVM::GEPArg gepIndices[2]{{0}, {0}};
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
      /*inbounds=*/true);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMComplexImagPtrOpLowering::matchAndRewrite(
    cir::ComplexImagPtrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto operandTy = mlir::cast<cir::PointerType>(op.getOperand().getType());
  auto resultLLVMTy = getTypeConverter()->convertType(op.getResult().getType());
  auto elementLLVMTy = getTypeConverter()->convertType(operandTy.getPointee());

  mlir::LLVM::GEPArg gepIndices[2]{{0}, {1}};
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      op, resultLLVMTy, elementLLVMTy, adaptor.getOperand(), gepIndices,
      /*inbounds=*/true);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMSwitchFlatOpLowering::matchAndRewrite(
    cir::SwitchFlatOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  llvm::SmallVector<mlir::APInt, 8> caseValues;
  if (op.getCaseValues()) {
    for (auto val : op.getCaseValues()) {
      auto intAttr = dyn_cast<cir::IntAttr>(val);
      caseValues.push_back(intAttr.getValue());
    }
  }

  llvm::SmallVector<mlir::Block *, 8> caseDestinations;
  llvm::SmallVector<mlir::ValueRange, 8> caseOperands;

  for (auto x : op.getCaseDestinations()) {
    caseDestinations.push_back(x);
  }

  for (auto x : op.getCaseOperands()) {
    caseOperands.push_back(x);
  }

  // Set switch op to branch to the newly created blocks.
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::LLVM::SwitchOp>(
      op, adaptor.getCondition(), op.getDefaultDestination(),
      op.getDefaultOperands(), caseValues, caseDestinations, caseOperands);
  return mlir::success();
}

/// Replace CIR global with a region initialized LLVM global and update
/// insertion point to the end of the initializer block.
void CIRToLLVMGlobalOpLowering::setupRegionInitializedLLVMGlobalOp(
    cir::GlobalOp op, mlir::ConversionPatternRewriter &rewriter) const {
  const auto llvmType = getTypeConverter()->convertType(op.getSymType());
  SmallVector<mlir::NamedAttribute> attributes;
  auto newGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
      op, llvmType, op.getConstant(), convertLinkage(op.getLinkage()),
      op.getSymName(), nullptr,
      /*alignment*/ op.getAlignment().value_or(0),
      /*addrSpace*/ getGlobalOpTargetAddrSpace(rewriter, typeConverter, op),
      /*dsoLocal*/ false, /*threadLocal*/ (bool)op.getTlsModelAttr(),
      /*comdat*/ mlir::SymbolRefAttr(), attributes);
  newGlobalOp.getRegion().push_back(new mlir::Block());
  rewriter.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());
}

mlir::LogicalResult CIRToLLVMGlobalOpLowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  // Fetch required values to create LLVM op.
  const auto llvmType = getTypeConverter()->convertType(op.getSymType());
  const auto isConst = op.getConstant();
  const auto isDsoLocal = op.getDsolocal();
  const auto linkage = convertLinkage(op.getLinkage());
  const auto symbol = op.getSymName();
  const auto loc = op.getLoc();
  std::optional<mlir::StringRef> section = op.getSection();
  std::optional<mlir::Attribute> init = op.getInitialValue();
  mlir::LLVM::VisibilityAttr visibility = mlir::LLVM::VisibilityAttr::get(
      getContext(), lowerCIRVisibilityToLLVMVisibility(
                        op.getGlobalVisibilityAttr().getValue()));

  SmallVector<mlir::NamedAttribute> attributes;
  if (section.has_value())
    attributes.push_back(rewriter.getNamedAttr(
        "section", rewriter.getStringAttr(section.value())));

  attributes.push_back(rewriter.getNamedAttr("visibility_", visibility));

  // Check for missing funcionalities.
  if (!init.has_value()) {
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, llvmType, isConst, linkage, symbol, mlir::Attribute(),
        /*alignment*/ 0,
        /*addrSpace*/ getGlobalOpTargetAddrSpace(rewriter, typeConverter, op),
        /*dsoLocal*/ isDsoLocal, /*threadLocal*/ (bool)op.getTlsModelAttr(),
        /*comdat*/ mlir::SymbolRefAttr(), attributes);
    return mlir::success();
  }

  // Initializer is a constant array: convert it to a compatible llvm init.
  if (auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(init.value())) {
    if (auto attr = mlir::dyn_cast<mlir::StringAttr>(constArr.getElts())) {
      llvm::SmallString<256> literal(attr.getValue());
      if (constArr.getTrailingZerosNum())
        literal.append(constArr.getTrailingZerosNum(), '\0');
      init = rewriter.getStringAttr(literal);
    } else if (auto attr =
                   mlir::dyn_cast<mlir::ArrayAttr>(constArr.getElts())) {
      // Failed to use a compact attribute as an initializer:
      // initialize elements individually.
      if (!(init = lowerConstArrayAttr(constArr, getTypeConverter()))) {
        setupRegionInitializedLLVMGlobalOp(op, rewriter);
        rewriter.create<mlir::LLVM::ReturnOp>(
            op->getLoc(),
            lowerCirAttrAsValue(op, constArr, rewriter, typeConverter));
        return mlir::success();
      }
    } else {
      op.emitError() << "unsupported lowering for #cir.const_array with value "
                     << constArr.getElts();
      return mlir::failure();
    }
  } else if (auto fltAttr = mlir::dyn_cast<cir::FPAttr>(init.value())) {
    // Initializer is a constant floating-point number: convert to MLIR
    // builtin constant.
    init = rewriter.getFloatAttr(llvmType, fltAttr.getValue());
  }
  // Initializer is a constant integer: convert to MLIR builtin constant.
  else if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(init.value())) {
    init = rewriter.getIntegerAttr(llvmType, intAttr.getValue());
  } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(init.value())) {
    init = rewriter.getBoolAttr(boolAttr.getValue());
  } else if (isa<cir::ZeroAttr, cir::ConstPtrAttr, cir::UndefAttr>(
                 init.value())) {
    // TODO(cir): once LLVM's dialect has proper equivalent attributes this
    // should be updated. For now, we use a custom op to initialize globals
    // to the appropriate value.
    setupRegionInitializedLLVMGlobalOp(op, rewriter);
    auto value = lowerCirAttrAsValue(op, init.value(), rewriter, typeConverter);
    rewriter.create<mlir::LLVM::ReturnOp>(loc, value);
    return mlir::success();
  } else if (auto dataMemberAttr =
                 mlir::dyn_cast<cir::DataMemberAttr>(init.value())) {
    assert(lowerMod && "lower module is not available");
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerMod->getCXXABI().lowerDataMemberConstant(
        dataMemberAttr, layout, *typeConverter);
    auto abiOp = mlir::cast<GlobalOp>(rewriter.clone(*op.getOperation()));
    abiOp.setInitialValueAttr(abiValue);
    abiOp.setSymType(abiValue.getType());
    rewriter.replaceOp(op, abiOp);
    return mlir::success();
  } else if (const auto structAttr =
                 mlir::dyn_cast<cir::ConstStructAttr>(init.value())) {
    setupRegionInitializedLLVMGlobalOp(op, rewriter);
    rewriter.create<mlir::LLVM::ReturnOp>(
        op->getLoc(),
        lowerCirAttrAsValue(op, structAttr, rewriter, typeConverter));
    return mlir::success();
  } else if (auto attr = mlir::dyn_cast<cir::GlobalViewAttr>(init.value())) {
    setupRegionInitializedLLVMGlobalOp(op, rewriter);
    rewriter.create<mlir::LLVM::ReturnOp>(
        loc, lowerCirAttrAsValue(op, attr, rewriter, typeConverter));
    return mlir::success();
  } else if (const auto vtableAttr =
                 mlir::dyn_cast<cir::VTableAttr>(init.value())) {
    setupRegionInitializedLLVMGlobalOp(op, rewriter);
    rewriter.create<mlir::LLVM::ReturnOp>(
        op->getLoc(),
        lowerCirAttrAsValue(op, vtableAttr, rewriter, typeConverter));
    return mlir::success();
  } else if (const auto typeinfoAttr =
                 mlir::dyn_cast<cir::TypeInfoAttr>(init.value())) {
    setupRegionInitializedLLVMGlobalOp(op, rewriter);
    rewriter.create<mlir::LLVM::ReturnOp>(
        op->getLoc(),
        lowerCirAttrAsValue(op, typeinfoAttr, rewriter, typeConverter));
    return mlir::success();
  } else {
    op.emitError() << "unsupported initializer '" << init.value() << "'";
    return mlir::failure();
  }

  // Rewrite op.
  auto llvmGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
      op, llvmType, isConst, linkage, symbol, init.value(),
      /*alignment*/ op.getAlignment().value_or(0),
      /*addrSpace*/ getGlobalOpTargetAddrSpace(rewriter, typeConverter, op),
      /*dsoLocal*/ false, /*threadLocal*/ (bool)op.getTlsModelAttr(),
      /*comdat*/ mlir::SymbolRefAttr(), attributes);

  auto mod = op->getParentOfType<mlir::ModuleOp>();
  if (op.getComdat())
    addComdat(llvmGlobalOp, comdatOp, rewriter, mod);

  return mlir::success();
}

void CIRToLLVMGlobalOpLowering::addComdat(mlir::LLVM::GlobalOp &op,
                                          mlir::LLVM::ComdatOp &comdatOp,
                                          mlir::OpBuilder &builder,
                                          mlir::ModuleOp &module) {
  StringRef comdatName("__llvm_comdat_globals");
  if (!comdatOp) {
    builder.setInsertionPointToStart(module.getBody());
    comdatOp =
        builder.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
  }
  builder.setInsertionPointToStart(&comdatOp.getBody().back());
  auto selectorOp = builder.create<mlir::LLVM::ComdatSelectorOp>(
      comdatOp.getLoc(), op.getSymName(), mlir::LLVM::comdat::Comdat::Any);
  op.setComdatAttr(mlir::SymbolRefAttr::get(
      builder.getContext(), comdatName,
      mlir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
}

mlir::LogicalResult CIRToLLVMUnaryOpLowering::matchAndRewrite(
    cir::UnaryOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(op.getType() == op.getInput().getType() &&
         "Unary operation's operand type and result type are different");
  mlir::Type type = op.getType();
  mlir::Type elementType = elementTypeIfVector(type);
  bool IsVector = mlir::isa<cir::VectorType>(type);
  auto llvmType = getTypeConverter()->convertType(type);
  auto loc = op.getLoc();

  // Integer unary operations: + - ~ ++ --
  if (mlir::isa<cir::IntType>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      assert(!IsVector && "++ not allowed on vector types");
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, mlir::IntegerAttr::get(llvmType, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmType,
                                                     adaptor.getInput(), One);
      return mlir::success();
    }
    case cir::UnaryOpKind::Dec: {
      assert(!IsVector && "-- not allowed on vector types");
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmType, mlir::IntegerAttr::get(llvmType, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType,
                                                     adaptor.getInput(), One);
      return mlir::success();
    }
    case cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Minus: {
      mlir::Value Zero;
      if (IsVector)
        Zero = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmType);
      else
        Zero = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, llvmType, mlir::IntegerAttr::get(llvmType, 0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType, Zero,
                                                     adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Not: {
      // bit-wise compliment operator, implemented as an XOR with -1.
      mlir::Value minusOne;
      if (IsVector) {
        // Creating a vector object with all -1 values is easier said than
        // done. It requires a series of insertelement ops.
        mlir::Type llvmElementType =
            getTypeConverter()->convertType(elementType);
        auto MinusOneInt = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, llvmElementType, mlir::IntegerAttr::get(llvmElementType, -1));
        minusOne = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmType);
        auto NumElements = mlir::dyn_cast<cir::VectorType>(type).getSize();
        for (uint64_t i = 0; i < NumElements; ++i) {
          mlir::Value indexValue = rewriter.create<mlir::LLVM::ConstantOp>(
              loc, rewriter.getI64Type(), i);
          minusOne = rewriter.create<mlir::LLVM::InsertElementOp>(
              loc, minusOne, MinusOneInt, indexValue);
        }
      } else {
        minusOne = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, llvmType, mlir::IntegerAttr::get(llvmType, -1));
      }
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(
          op, llvmType, adaptor.getInput(), minusOne);
      return mlir::success();
    }
    }
  }

  // Floating point unary operations: + - ++ --
  if (mlir::isa<cir::CIRFPTypeInterface>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Inc: {
      assert(!IsVector && "++ not allowed on vector types");
      auto oneAttr = rewriter.getFloatAttr(llvmType, 1.0);
      auto oneConst =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, oneAttr);
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, oneConst,
                                                      adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Dec: {
      assert(!IsVector && "-- not allowed on vector types");
      auto negOneAttr = rewriter.getFloatAttr(llvmType, -1.0);
      auto negOneConst =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmType, negOneAttr);
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmType, negOneConst,
                                                      adaptor.getInput());
      return mlir::success();
    }
    case cir::UnaryOpKind::Plus:
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    case cir::UnaryOpKind::Minus: {
      rewriter.replaceOpWithNewOp<mlir::LLVM::FNegOp>(op, llvmType,
                                                      adaptor.getInput());
      return mlir::success();
    }
    default:
      return op.emitError()
             << "Unknown floating-point unary operation during CIR lowering";
    }
  }

  // Boolean unary operations: ! only. (For all others, the operand has
  // already been promoted to int.)
  if (mlir::isa<cir::BoolType>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Not:
      assert(!IsVector && "NYI: op! on vector mask");
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(
          op, llvmType, adaptor.getInput(),
          rewriter.create<mlir::LLVM::ConstantOp>(
              loc, llvmType, mlir::IntegerAttr::get(llvmType, 1)));
      return mlir::success();
    default:
      return op.emitError()
             << "Unknown boolean unary operation during CIR lowering";
    }
  }

  // Pointer unary operations: + only.  (++ and -- of pointers are implemented
  // with cir.ptr_stride, not cir.unary.)
  if (mlir::isa<cir::PointerType>(elementType)) {
    switch (op.getKind()) {
    case cir::UnaryOpKind::Plus:
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    default:
      op.emitError() << "Unknown pointer unary operation during CIR lowering";
      return mlir::failure();
    }
  }

  return op.emitError() << "Unary operation has unsupported type: "
                        << elementType;
}

mlir::LLVM::IntegerOverflowFlags
CIRToLLVMBinOpLowering::getIntOverflowFlag(cir::BinOp op) const {
  if (op.getNoUnsignedWrap())
    return mlir::LLVM::IntegerOverflowFlags::nuw;

  if (op.getNoSignedWrap())
    return mlir::LLVM::IntegerOverflowFlags::nsw;

  return mlir::LLVM::IntegerOverflowFlags::none;
}

mlir::LogicalResult CIRToLLVMBinOpLowering::matchAndRewrite(
    cir::BinOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert((adaptor.getLhs().getType() == adaptor.getRhs().getType()) &&
         "inconsistent operands' types not supported yet");

  mlir::Type type = op.getRhs().getType();
  assert((mlir::isa<cir::IntType, cir::CIRFPTypeInterface, cir::VectorType,
                    mlir::IntegerType>(type)) &&
         "operand type not supported yet");

  auto llvmTy = getTypeConverter()->convertType(op.getType());
  auto rhs = adaptor.getRhs();
  auto lhs = adaptor.getLhs();

  type = elementTypeIfVector(type);

  switch (op.getKind()) {
  case cir::BinOpKind::Add:
    if (mlir::isa<cir::IntType, mlir::IntegerType>(type))
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmTy, lhs, rhs,
                                                     getIntOverflowFlag(op));
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmTy, lhs, rhs);
    break;
  case cir::BinOpKind::Sub:
    if (mlir::isa<cir::IntType, mlir::IntegerType>(type))
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmTy, lhs, rhs,
                                                     getIntOverflowFlag(op));
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(op, llvmTy, lhs, rhs);
    break;
  case cir::BinOpKind::Mul:
    if (mlir::isa<cir::IntType, mlir::IntegerType>(type))
      rewriter.replaceOpWithNewOp<mlir::LLVM::MulOp>(op, llvmTy, lhs, rhs,
                                                     getIntOverflowFlag(op));
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(op, llvmTy, lhs, rhs);
    break;
  case cir::BinOpKind::Div:
    if (mlir::isa<cir::IntType, mlir::IntegerType>(type)) {
      auto isUnsigned = mlir::isa<cir::IntType>(type)
                            ? mlir::cast<cir::IntType>(type).isUnsigned()
                            : mlir::cast<mlir::IntegerType>(type).isUnsigned();
      if (isUnsigned)
        rewriter.replaceOpWithNewOp<mlir::LLVM::UDivOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::SDivOp>(op, llvmTy, lhs, rhs);
    } else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FDivOp>(op, llvmTy, lhs, rhs);
    break;
  case cir::BinOpKind::Rem:
    if (mlir::isa<cir::IntType, mlir::IntegerType>(type)) {
      auto isUnsigned = mlir::isa<cir::IntType>(type)
                            ? mlir::cast<cir::IntType>(type).isUnsigned()
                            : mlir::cast<mlir::IntegerType>(type).isUnsigned();
      if (isUnsigned)
        rewriter.replaceOpWithNewOp<mlir::LLVM::URemOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::SRemOp>(op, llvmTy, lhs, rhs);
    } else
      rewriter.replaceOpWithNewOp<mlir::LLVM::FRemOp>(op, llvmTy, lhs, rhs);
    break;
  case cir::BinOpKind::And:
    rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(op, llvmTy, lhs, rhs);
    break;
  case cir::BinOpKind::Or:
    rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, llvmTy, lhs, rhs);
    break;
  case cir::BinOpKind::Xor:
    rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, llvmTy, lhs, rhs);
    break;
  }

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBinOpOverflowOpLowering::matchAndRewrite(
    cir::BinOpOverflowOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto arithKind = op.getKind();
  auto operandTy = op.getLhs().getType();
  auto resultTy = op.getResult().getType();

  auto encompassedTyInfo = computeEncompassedTypeWidth(operandTy, resultTy);
  auto encompassedLLVMTy = rewriter.getIntegerType(encompassedTyInfo.width);

  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (operandTy.getWidth() < encompassedTyInfo.width) {
    if (operandTy.isSigned()) {
      lhs = rewriter.create<mlir::LLVM::SExtOp>(loc, encompassedLLVMTy, lhs);
      rhs = rewriter.create<mlir::LLVM::SExtOp>(loc, encompassedLLVMTy, rhs);
    } else {
      lhs = rewriter.create<mlir::LLVM::ZExtOp>(loc, encompassedLLVMTy, lhs);
      rhs = rewriter.create<mlir::LLVM::ZExtOp>(loc, encompassedLLVMTy, rhs);
    }
  }

  auto intrinName = getLLVMIntrinName(arithKind, encompassedTyInfo.sign,
                                      encompassedTyInfo.width);
  auto intrinNameAttr = mlir::StringAttr::get(op.getContext(), intrinName);

  auto overflowLLVMTy = rewriter.getI1Type();
  auto intrinRetTy = mlir::LLVM::LLVMStructType::getLiteral(
      rewriter.getContext(), {encompassedLLVMTy, overflowLLVMTy});

  auto callLLVMIntrinOp = rewriter.create<mlir::LLVM::CallIntrinsicOp>(
      loc, intrinRetTy, intrinNameAttr, mlir::ValueRange{lhs, rhs});
  auto intrinRet = callLLVMIntrinOp.getResult(0);

  auto result = rewriter
                    .create<mlir::LLVM::ExtractValueOp>(loc, intrinRet,
                                                        ArrayRef<int64_t>{0})
                    .getResult();
  auto overflow = rewriter
                      .create<mlir::LLVM::ExtractValueOp>(loc, intrinRet,
                                                          ArrayRef<int64_t>{1})
                      .getResult();

  if (resultTy.getWidth() < encompassedTyInfo.width) {
    auto resultLLVMTy = getTypeConverter()->convertType(resultTy);
    auto truncResult =
        rewriter.create<mlir::LLVM::TruncOp>(loc, resultLLVMTy, result);

    // Extend the truncated result back to the encompassing type to check for
    // any overflows during the truncation.
    mlir::Value truncResultExt;
    if (resultTy.isSigned())
      truncResultExt = rewriter.create<mlir::LLVM::SExtOp>(
          loc, encompassedLLVMTy, truncResult);
    else
      truncResultExt = rewriter.create<mlir::LLVM::ZExtOp>(
          loc, encompassedLLVMTy, truncResult);
    auto truncOverflow = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, truncResultExt, result);

    result = truncResult;
    overflow = rewriter.create<mlir::LLVM::OrOp>(loc, overflow, truncOverflow);
  }

  auto boolLLVMTy = getTypeConverter()->convertType(op.getOverflow().getType());
  if (boolLLVMTy != rewriter.getI1Type())
    overflow = rewriter.create<mlir::LLVM::ZExtOp>(loc, boolLLVMTy, overflow);

  rewriter.replaceOp(op, mlir::ValueRange{result, overflow});

  return mlir::success();
}

std::string CIRToLLVMBinOpOverflowOpLowering::getLLVMIntrinName(
    cir::BinOpOverflowKind opKind, bool isSigned, unsigned width) {
  // The intrinsic name is `@llvm.{s|u}{opKind}.with.overflow.i{width}`

  std::string name = "llvm.";

  if (isSigned)
    name.push_back('s');
  else
    name.push_back('u');

  switch (opKind) {
  case cir::BinOpOverflowKind::Add:
    name.append("add.");
    break;
  case cir::BinOpOverflowKind::Sub:
    name.append("sub.");
    break;
  case cir::BinOpOverflowKind::Mul:
    name.append("mul.");
    break;
  }

  name.append("with.overflow.i");
  name.append(std::to_string(width));

  return name;
}

CIRToLLVMBinOpOverflowOpLowering::EncompassedTypeInfo
CIRToLLVMBinOpOverflowOpLowering::computeEncompassedTypeWidth(
    cir::IntType operandTy, cir::IntType resultTy) {
  auto sign = operandTy.getIsSigned() || resultTy.getIsSigned();
  auto width = std::max(operandTy.getWidth() + (sign && operandTy.isUnsigned()),
                        resultTy.getWidth() + (sign && resultTy.isUnsigned()));
  return {sign, width};
}

mlir::LogicalResult CIRToLLVMShiftOpLowering::matchAndRewrite(
    cir::ShiftOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto cirAmtTy = mlir::dyn_cast<cir::IntType>(op.getAmount().getType());
  auto cirValTy = mlir::dyn_cast<cir::IntType>(op.getValue().getType());

  // Operands could also be vector type
  auto cirAmtVTy = mlir::dyn_cast<cir::VectorType>(op.getAmount().getType());
  auto cirValVTy = mlir::dyn_cast<cir::VectorType>(op.getValue().getType());
  auto llvmTy = getTypeConverter()->convertType(op.getType());
  mlir::Value amt = adaptor.getAmount();
  mlir::Value val = adaptor.getValue();

  assert(((cirValTy && cirAmtTy) || (cirAmtVTy && cirValVTy)) &&
         "shift input type must be integer or vector type, otherwise NYI");

  assert((cirValTy == op.getType() || cirValVTy == op.getType()) &&
         "inconsistent operands' types NYI");

  // Ensure shift amount is the same type as the value. Some undefined
  // behavior might occur in the casts below as per [C99 6.5.7.3].
  // Vector type shift amount needs no cast as type consistency is expected to
  // be already be enforced at CIRGen.
  if (cirAmtTy)
    amt = getLLVMIntCast(rewriter, amt, mlir::cast<mlir::IntegerType>(llvmTy),
                         !cirAmtTy.isSigned(), cirAmtTy.getWidth(),
                         cirValTy.getWidth());

  // Lower to the proper LLVM shift operation.
  if (op.getIsShiftleft())
    rewriter.replaceOpWithNewOp<mlir::LLVM::ShlOp>(op, llvmTy, val, amt);
  else {
    bool isUnSigned =
        cirValTy ? !cirValTy.isSigned()
                 : !mlir::cast<cir::IntType>(cirValVTy.getEltType()).isSigned();
    if (isUnSigned)
      rewriter.replaceOpWithNewOp<mlir::LLVM::LShrOp>(op, llvmTy, val, amt);
    else
      rewriter.replaceOpWithNewOp<mlir::LLVM::AShrOp>(op, llvmTy, val, amt);
  }

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCmpOpLowering::matchAndRewrite(
    cir::CmpOp cmpOp, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto type = cmpOp.getLhs().getType();
  mlir::Value llResult;

  // Lower to LLVM comparison op.
  // if (auto intTy = mlir::dyn_cast<cir::IntType>(type)) {
  if (mlir::isa<cir::IntType, mlir::IntegerType>(type)) {
    auto isSigned = mlir::isa<cir::IntType>(type)
                        ? mlir::cast<cir::IntType>(type).isSigned()
                        : mlir::cast<mlir::IntegerType>(type).isSigned();
    auto kind = convertCmpKindToICmpPredicate(cmpOp.getKind(), isSigned);
    llResult = rewriter.create<mlir::LLVM::ICmpOp>(
        cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
  } else if (auto ptrTy = mlir::dyn_cast<cir::PointerType>(type)) {
    auto kind = convertCmpKindToICmpPredicate(cmpOp.getKind(),
                                              /* isSigned=*/false);
    llResult = rewriter.create<mlir::LLVM::ICmpOp>(
        cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
  } else if (mlir::isa<cir::CIRFPTypeInterface>(type)) {
    auto kind = convertCmpKindToFCmpPredicate(cmpOp.getKind());
    llResult = rewriter.create<mlir::LLVM::FCmpOp>(
        cmpOp.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
  } else {
    return cmpOp.emitError() << "unsupported type for CmpOp: " << type;
  }

  // LLVM comparison ops return i1, but cir::CmpOp returns the same type as
  // the LHS value. Since this return value can be used later, we need to
  // restore the type with the extension below.
  auto llResultTy = getTypeConverter()->convertType(cmpOp.getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, llResultTy, llResult);

  return mlir::success();
}

mlir::LLVM::CallIntrinsicOp
createCallLLVMIntrinsicOp(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Location loc, const llvm::Twine &intrinsicName,
                          mlir::Type resultTy, mlir::ValueRange operands) {
  auto intrinsicNameAttr =
      mlir::StringAttr::get(rewriter.getContext(), intrinsicName);
  return rewriter.create<mlir::LLVM::CallIntrinsicOp>(
      loc, resultTy, intrinsicNameAttr, operands);
}

mlir::LLVM::CallIntrinsicOp replaceOpWithCallLLVMIntrinsicOp(
    mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
    const llvm::Twine &intrinsicName, mlir::Type resultTy,
    mlir::ValueRange operands) {
  auto callIntrinOp = createCallLLVMIntrinsicOp(
      rewriter, op->getLoc(), intrinsicName, resultTy, operands);
  rewriter.replaceOp(op, callIntrinOp.getOperation());
  return callIntrinOp;
}

mlir::LogicalResult CIRToLLVMLLVMIntrinsicCallOpLowering::matchAndRewrite(
    cir::LLVMIntrinsicCallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type llvmResTy =
      getTypeConverter()->convertType(op->getResultTypes()[0]);
  if (!llvmResTy)
    return op.emitError("expected LLVM result type");
  StringRef name = op.getIntrinsicName();
  // Some llvm intrinsics require ElementType attribute to be attached to
  // the argument of pointer type. That prevents us from generating LLVM IR
  // because from LLVM dialect, we have LLVM IR like the below which fails
  // LLVM IR verification.
  // %3 = call i64 @llvm.aarch64.ldxr.p0(ptr %2)
  // The expected LLVM IR should be like
  // %3 = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i32) %2)
  // TODO(cir): MLIR LLVM dialect should handle this part as CIR has no way
  // to set LLVM IR attribute.
  assert(!cir::MissingFeatures::llvmIntrinsicElementTypeSupport());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm." + name, llvmResTy,
                                   adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAssumeOpLowering::matchAndRewrite(
    cir::AssumeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto cond = rewriter.create<mlir::LLVM::TruncOp>(
      op.getLoc(), rewriter.getI1Type(), adaptor.getPredicate());
  rewriter.replaceOpWithNewOp<mlir::LLVM::AssumeOp>(op, cond);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAssumeAlignedOpLowering::matchAndRewrite(
    cir::AssumeAlignedOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  SmallVector<mlir::Value, 3> opBundleArgs{adaptor.getPointer()};

  auto alignment = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(), op.getAlignment());
  opBundleArgs.push_back(alignment);

  if (mlir::Value offset = adaptor.getOffset())
    opBundleArgs.push_back(offset);

  auto cond = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 1);
  rewriter.create<mlir::LLVM::AssumeOp>(op.getLoc(), cond, "align",
                                        opBundleArgs);
  rewriter.replaceAllUsesWith(op, op.getPointer());
  rewriter.eraseOp(op);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAssumeSepStorageOpLowering::matchAndRewrite(
    cir::AssumeSepStorageOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto cond = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 1);
  rewriter.replaceOpWithNewOp<mlir::LLVM::AssumeOp>(
      op, cond, "separate_storage",
      mlir::ValueRange{adaptor.getPtr1(), adaptor.getPtr2()});
  return mlir::success();
}

mlir::Value createLLVMBitOp(mlir::Location loc,
                            const llvm::Twine &llvmIntrinBaseName,
                            mlir::Type resultTy, mlir::Value operand,
                            std::optional<bool> poisonZeroInputFlag,
                            mlir::ConversionPatternRewriter &rewriter) {
  auto operandIntTy = mlir::cast<mlir::IntegerType>(operand.getType());
  auto resultIntTy = mlir::cast<mlir::IntegerType>(resultTy);

  std::string llvmIntrinName =
      llvmIntrinBaseName.concat(".i")
          .concat(std::to_string(operandIntTy.getWidth()))
          .str();

  // Note that LLVM intrinsic calls to bit intrinsics have the same type as the
  // operand.
  mlir::LLVM::CallIntrinsicOp op;
  if (poisonZeroInputFlag.has_value()) {
    auto poisonZeroInputValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), static_cast<int64_t>(*poisonZeroInputFlag));
    op = createCallLLVMIntrinsicOp(rewriter, loc, llvmIntrinName,
                                   operand.getType(),
                                   {operand, poisonZeroInputValue});
  } else {
    op = createCallLLVMIntrinsicOp(rewriter, loc, llvmIntrinName,
                                   operand.getType(), operand);
  }

  return getLLVMIntCast(
      rewriter, op->getResult(0), mlir::cast<mlir::IntegerType>(resultTy),
      /*isUnsigned=*/true, operandIntTy.getWidth(), resultIntTy.getWidth());
}

mlir::LogicalResult CIRToLLVMBitClrsbOpLowering::matchAndRewrite(
    cir::BitClrsbOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), adaptor.getInput().getType(), 0);
  auto isNeg = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(),
      mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                         mlir::LLVM::ICmpPredicate::slt),
      adaptor.getInput(), zero);

  auto negOne = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), adaptor.getInput().getType(), -1);
  auto flipped = rewriter.create<mlir::LLVM::XOrOp>(op.getLoc(),
                                                    adaptor.getInput(), negOne);

  auto select = rewriter.create<mlir::LLVM::SelectOp>(
      op.getLoc(), isNeg, flipped, adaptor.getInput());

  auto resTy = getTypeConverter()->convertType(op.getType());
  auto clz = createLLVMBitOp(op.getLoc(), "llvm.ctlz", resTy, select,
                             /*poisonZeroInputFlag=*/false, rewriter);

  auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
  auto res = rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), clz, one);
  rewriter.replaceOp(op, res);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMObjSizeOpLowering::matchAndRewrite(
    cir::ObjSizeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmResTy = getTypeConverter()->convertType(op.getType());
  auto loc = op->getLoc();

  cir::SizeInfoType kindInfo = op.getKind();
  auto falseValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI1Type(), false);
  auto trueValue =
      rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI1Type(), true);

  replaceOpWithCallLLVMIntrinsicOp(
      rewriter, op, "llvm.objectsize", llvmResTy,
      mlir::ValueRange{adaptor.getPtr(),
                       kindInfo == cir::SizeInfoType::max ? falseValue
                                                          : trueValue,
                       trueValue, op.getDynamic() ? trueValue : falseValue});

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitClzOpLowering::matchAndRewrite(
    cir::BitClzOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp =
      createLLVMBitOp(op.getLoc(), "llvm.ctlz", resTy, adaptor.getInput(),
                      /*poisonZeroInputFlag=*/true, rewriter);
  rewriter.replaceOp(op, llvmOp);
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitCtzOpLowering::matchAndRewrite(
    cir::BitCtzOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp =
      createLLVMBitOp(op.getLoc(), "llvm.cttz", resTy, adaptor.getInput(),
                      /*poisonZeroInputFlag=*/true, rewriter);
  rewriter.replaceOp(op, llvmOp);
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitFfsOpLowering::matchAndRewrite(
    cir::BitFfsOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto ctz =
      createLLVMBitOp(op.getLoc(), "llvm.cttz", resTy, adaptor.getInput(),
                      /*poisonZeroInputFlag=*/false, rewriter);

  auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
  auto ctzAddOne = rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), ctz, one);

  auto zeroInputTy = rewriter.create<mlir::LLVM::ConstantOp>(
      op.getLoc(), adaptor.getInput().getType(), 0);
  auto isZero = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(),
      mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(),
                                         mlir::LLVM::ICmpPredicate::eq),
      adaptor.getInput(), zeroInputTy);

  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 0);
  auto res = rewriter.create<mlir::LLVM::SelectOp>(op.getLoc(), isZero, zero,
                                                   ctzAddOne);
  rewriter.replaceOp(op, res);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitParityOpLowering::matchAndRewrite(
    cir::BitParityOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto popcnt =
      createLLVMBitOp(op.getLoc(), "llvm.ctpop", resTy, adaptor.getInput(),
                      /*poisonZeroInputFlag=*/std::nullopt, rewriter);

  auto one = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), resTy, 1);
  auto popcntMod2 =
      rewriter.create<mlir::LLVM::AndOp>(op.getLoc(), popcnt, one);
  rewriter.replaceOp(op, popcntMod2);

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMBitPopcountOpLowering::matchAndRewrite(
    cir::BitPopcountOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = getTypeConverter()->convertType(op.getType());
  auto llvmOp =
      createLLVMBitOp(op.getLoc(), "llvm.ctpop", resTy, adaptor.getInput(),
                      /*poisonZeroInputFlag=*/std::nullopt, rewriter);
  rewriter.replaceOp(op, llvmOp);
  return mlir::LogicalResult::success();
}

mlir::LLVM::AtomicOrdering getLLVMAtomicOrder(cir::MemOrder memo) {
  switch (memo) {
  case cir::MemOrder::Relaxed:
    return mlir::LLVM::AtomicOrdering::monotonic;
  case cir::MemOrder::Consume:
  case cir::MemOrder::Acquire:
    return mlir::LLVM::AtomicOrdering::acquire;
  case cir::MemOrder::Release:
    return mlir::LLVM::AtomicOrdering::release;
  case cir::MemOrder::AcquireRelease:
    return mlir::LLVM::AtomicOrdering::acq_rel;
  case cir::MemOrder::SequentiallyConsistent:
    return mlir::LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("shouldn't get here");
}

mlir::LogicalResult CIRToLLVMAtomicCmpXchgLowering::matchAndRewrite(
    cir::AtomicCmpXchg op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto expected = adaptor.getExpected();
  auto desired = adaptor.getDesired();

  // FIXME: add syncscope.
  auto cmpxchg = rewriter.create<mlir::LLVM::AtomicCmpXchgOp>(
      op.getLoc(), adaptor.getPtr(), expected, desired,
      getLLVMAtomicOrder(adaptor.getSuccOrder()),
      getLLVMAtomicOrder(adaptor.getFailOrder()));
  cmpxchg.setWeak(adaptor.getWeak());
  cmpxchg.setVolatile_(adaptor.getIsVolatile());

  // Check result and apply stores accordingly.
  auto old = rewriter.create<mlir::LLVM::ExtractValueOp>(
      op.getLoc(), cmpxchg.getResult(), 0);
  auto cmp = rewriter.create<mlir::LLVM::ExtractValueOp>(
      op.getLoc(), cmpxchg.getResult(), 1);

  auto extCmp = rewriter.create<mlir::LLVM::ZExtOp>(op.getLoc(),
                                                    rewriter.getI8Type(), cmp);
  rewriter.replaceOp(op, {old, extCmp});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAtomicXchgLowering::matchAndRewrite(
    cir::AtomicXchg op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // FIXME: add syncscope.
  auto llvmOrder = getLLVMAtomicOrder(adaptor.getMemOrder());
  rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
      op, mlir::LLVM::AtomicBinOp::xchg, adaptor.getPtr(), adaptor.getVal(),
      llvmOrder);
  return mlir::success();
}

mlir::Value CIRToLLVMAtomicFetchLowering::buildPostOp(
    cir::AtomicFetch op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter, mlir::Value rmwVal,
    bool isInt) const {
  SmallVector<mlir::Value> atomicOperands = {rmwVal, adaptor.getVal()};
  SmallVector<mlir::Type> atomicResTys = {rmwVal.getType()};
  return rewriter
      .create(op.getLoc(),
              rewriter.getStringAttr(getLLVMBinop(op.getBinop(), isInt)),
              atomicOperands, atomicResTys, {})
      ->getResult(0);
}

mlir::Value CIRToLLVMAtomicFetchLowering::buildMinMaxPostOp(
    cir::AtomicFetch op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter, mlir::Value rmwVal,
    bool isSigned) const {
  auto loc = op.getLoc();
  mlir::LLVM::ICmpPredicate pred;
  if (op.getBinop() == cir::AtomicFetchKind::Max) {
    pred = isSigned ? mlir::LLVM::ICmpPredicate::sgt
                    : mlir::LLVM::ICmpPredicate::ugt;
  } else { // Min
    pred = isSigned ? mlir::LLVM::ICmpPredicate::slt
                    : mlir::LLVM::ICmpPredicate::ult;
  }

  auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicateAttr::get(rewriter.getContext(), pred),
      rmwVal, adaptor.getVal());
  return rewriter.create<mlir::LLVM::SelectOp>(loc, cmp, rmwVal,
                                               adaptor.getVal());
}

llvm::StringLiteral
CIRToLLVMAtomicFetchLowering::getLLVMBinop(cir::AtomicFetchKind k,
                                           bool isInt) const {
  switch (k) {
  case cir::AtomicFetchKind::Add:
    return isInt ? mlir::LLVM::AddOp::getOperationName()
                 : mlir::LLVM::FAddOp::getOperationName();
  case cir::AtomicFetchKind::Sub:
    return isInt ? mlir::LLVM::SubOp::getOperationName()
                 : mlir::LLVM::FSubOp::getOperationName();
  case cir::AtomicFetchKind::And:
    return mlir::LLVM::AndOp::getOperationName();
  case cir::AtomicFetchKind::Xor:
    return mlir::LLVM::XOrOp::getOperationName();
  case cir::AtomicFetchKind::Or:
    return mlir::LLVM::OrOp::getOperationName();
  case cir::AtomicFetchKind::Nand:
    // There's no nand binop in LLVM, this is later fixed with a not.
    return mlir::LLVM::AndOp::getOperationName();
  case cir::AtomicFetchKind::Max:
  case cir::AtomicFetchKind::Min:
    llvm_unreachable("handled in buildMinMaxPostOp");
  }
  llvm_unreachable("Unknown atomic fetch opcode");
}

mlir::LLVM::AtomicBinOp CIRToLLVMAtomicFetchLowering::getLLVMAtomicBinOp(
    cir::AtomicFetchKind k, bool isInt, bool isSignedInt) const {
  switch (k) {
  case cir::AtomicFetchKind::Add:
    return isInt ? mlir::LLVM::AtomicBinOp::add : mlir::LLVM::AtomicBinOp::fadd;
  case cir::AtomicFetchKind::Sub:
    return isInt ? mlir::LLVM::AtomicBinOp::sub : mlir::LLVM::AtomicBinOp::fsub;
  case cir::AtomicFetchKind::And:
    return mlir::LLVM::AtomicBinOp::_and;
  case cir::AtomicFetchKind::Xor:
    return mlir::LLVM::AtomicBinOp::_xor;
  case cir::AtomicFetchKind::Or:
    return mlir::LLVM::AtomicBinOp::_or;
  case cir::AtomicFetchKind::Nand:
    return mlir::LLVM::AtomicBinOp::nand;
  case cir::AtomicFetchKind::Max: {
    if (!isInt)
      return mlir::LLVM::AtomicBinOp::fmax;
    return isSignedInt ? mlir::LLVM::AtomicBinOp::max
                       : mlir::LLVM::AtomicBinOp::umax;
  }
  case cir::AtomicFetchKind::Min: {
    if (!isInt)
      return mlir::LLVM::AtomicBinOp::fmin;
    return isSignedInt ? mlir::LLVM::AtomicBinOp::min
                       : mlir::LLVM::AtomicBinOp::umin;
  }
  }
  llvm_unreachable("Unknown atomic fetch opcode");
}

mlir::LogicalResult CIRToLLVMAtomicFetchLowering::matchAndRewrite(
    cir::AtomicFetch op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  bool isInt, isSignedInt = false; // otherwise it's float.
  if (auto intTy = mlir::dyn_cast<cir::IntType>(op.getVal().getType())) {
    isInt = true;
    isSignedInt = intTy.isSigned();
  } else if (mlir::isa<cir::SingleType, cir::DoubleType>(op.getVal().getType()))
    isInt = false;
  else {
    return op.emitError() << "Unsupported type: " << adaptor.getVal().getType();
  }

  // FIXME: add syncscope.
  auto llvmOrder = getLLVMAtomicOrder(adaptor.getMemOrder());
  auto llvmBinOpc = getLLVMAtomicBinOp(op.getBinop(), isInt, isSignedInt);
  auto rmwVal = rewriter.create<mlir::LLVM::AtomicRMWOp>(
      op.getLoc(), llvmBinOpc, adaptor.getPtr(), adaptor.getVal(), llvmOrder);

  mlir::Value result = rmwVal.getRes();
  if (!op.getFetchFirst()) {
    if (op.getBinop() == cir::AtomicFetchKind::Max ||
        op.getBinop() == cir::AtomicFetchKind::Min)
      result = buildMinMaxPostOp(op, adaptor, rewriter, rmwVal.getRes(),
                                 isSignedInt);
    else
      result = buildPostOp(op, adaptor, rewriter, rmwVal.getRes(), isInt);

    // Compensate lack of nand binop in LLVM IR.
    if (op.getBinop() == cir::AtomicFetchKind::Nand) {
      auto negOne = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), result.getType(), -1);
      result = rewriter.create<mlir::LLVM::XOrOp>(op.getLoc(), result, negOne);
    }
  }

  rewriter.replaceOp(op, result);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMByteswapOpLowering::matchAndRewrite(
    cir::ByteswapOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Note that LLVM intrinsic calls to @llvm.bswap.i* have the same type as
  // the operand.

  auto resTy = mlir::cast<mlir::IntegerType>(
      getTypeConverter()->convertType(op.getType()));

  std::string llvmIntrinName = "llvm.bswap.i";
  llvmIntrinName.append(std::to_string(resTy.getWidth()));

  rewriter.replaceOpWithNewOp<mlir::LLVM::ByteSwapOp>(op, adaptor.getInput());

  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMRotateOpLowering::matchAndRewrite(
    cir::RotateOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Note that LLVM intrinsic calls to @llvm.fsh{r,l}.i* have the same type as
  // the operand.
  auto src = adaptor.getSrc();
  if (op.getLeft())
    rewriter.replaceOpWithNewOp<mlir::LLVM::FshlOp>(op, src, src,
                                                    adaptor.getAmt());
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::FshrOp>(op, src, src,
                                                    adaptor.getAmt());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMSelectOpLowering::matchAndRewrite(
    cir::SelectOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto getConstantBool = [](mlir::Value value) -> std::optional<bool> {
    auto definingOp =
        mlir::dyn_cast_if_present<cir::ConstantOp>(value.getDefiningOp());
    if (!definingOp)
      return std::nullopt;

    auto constValue = mlir::dyn_cast<cir::BoolAttr>(definingOp.getValue());
    if (!constValue)
      return std::nullopt;

    return constValue.getValue();
  };

  // Two special cases in the LLVMIR codegen of select op:
  // - select %0, %1, false => and %0, %1
  // - select %0, true, %1 => or %0, %1
  auto trueValue = op.getTrueValue();
  auto falseValue = op.getFalseValue();
  if (mlir::isa<cir::BoolType>(trueValue.getType())) {
    if (std::optional<bool> falseValueBool = getConstantBool(falseValue);
        falseValueBool.has_value() && !*falseValueBool) {
      // select %0, %1, false => and %0, %1
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(op, adaptor.getCondition(),
                                                     adaptor.getTrueValue());
      return mlir::success();
    }
    if (std::optional<bool> trueValueBool = getConstantBool(trueValue);
        trueValueBool.has_value() && *trueValueBool) {
      // select %0, true, %1 => or %0, %1
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, adaptor.getCondition(),
                                                    adaptor.getFalseValue());
      return mlir::success();
    }
  }

  auto llvmCondition = rewriter.create<mlir::LLVM::TruncOp>(
      op.getLoc(), mlir::IntegerType::get(op->getContext(), 1),
      adaptor.getCondition());
  rewriter.replaceOpWithNewOp<mlir::LLVM::SelectOp>(
      op, llvmCondition, adaptor.getTrueValue(), adaptor.getFalseValue());

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMBrOpLowering::matchAndRewrite(
    cir::BrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, adaptor.getOperands(),
                                                op.getDest());
  return mlir::LogicalResult::success();
}

mlir::LogicalResult CIRToLLVMGetMemberOpLowering::matchAndRewrite(
    cir::GetMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llResTy = getTypeConverter()->convertType(op.getType());
  const auto structTy =
      mlir::cast<cir::StructType>(op.getAddrTy().getPointee());
  assert(structTy && "expected struct type");

  switch (structTy.getKind()) {
  case cir::StructType::Struct:
  case cir::StructType::Class: {
    // Since the base address is a pointer to an aggregate, the first offset
    // is always zero. The second offset tell us which member it will access.
    llvm::SmallVector<mlir::LLVM::GEPArg, 2> offset{0, op.getIndex()};
    const auto elementTy = getTypeConverter()->convertType(structTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, llResTy, elementTy,
                                                   adaptor.getAddr(), offset);
    return mlir::success();
  }
  case cir::StructType::Union:
    // Union members share the address space, so we just need a bitcast to
    // conform to type-checking.
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, llResTy,
                                                       adaptor.getAddr());
    return mlir::success();
  }
}

mlir::LogicalResult CIRToLLVMGetRuntimeMemberOpLowering::matchAndRewrite(
    cir::GetRuntimeMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(lowerMod && "lowering module is not available");
  mlir::Type llvmResTy = getTypeConverter()->convertType(op.getType());
  mlir::Operation *llvmOp = lowerMod->getCXXABI().lowerGetRuntimeMember(
      op, llvmResTy, adaptor.getAddr(), adaptor.getMember(), rewriter);
  rewriter.replaceOp(op, llvmOp);
  return mlir::success();
}

uint64_t CIRToLLVMPtrDiffOpLowering::getTypeSize(mlir::Type type,
                                                 mlir::Operation &op) const {
  mlir::DataLayout layout(op.getParentOfType<mlir::ModuleOp>());
  // For LLVM purposes we treat void as u8.
  if (isa<cir::VoidType>(type))
    type = cir::IntType::get(type.getContext(), 8, /*isSigned=*/false);
  return llvm::divideCeil(layout.getTypeSizeInBits(type), 8);
}

mlir::LogicalResult CIRToLLVMPtrDiffOpLowering::matchAndRewrite(
    cir::PtrDiffOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto dstTy = mlir::cast<cir::IntType>(op.getType());
  auto llvmDstTy = getTypeConverter()->convertType(dstTy);

  auto lhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                     adaptor.getLhs());
  auto rhs = rewriter.create<mlir::LLVM::PtrToIntOp>(op.getLoc(), llvmDstTy,
                                                     adaptor.getRhs());

  auto diff =
      rewriter.create<mlir::LLVM::SubOp>(op.getLoc(), llvmDstTy, lhs, rhs);

  auto ptrTy = mlir::cast<cir::PointerType>(op.getLhs().getType());
  auto typeSize = getTypeSize(ptrTy.getPointee(), *op);

  // Avoid silly division by 1.
  auto resultVal = diff.getResult();
  if (typeSize != 1) {
    auto typeSizeVal = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), llvmDstTy, mlir::IntegerAttr::get(llvmDstTy, typeSize));

    if (dstTy.isUnsigned())
      resultVal = rewriter.create<mlir::LLVM::UDivOp>(op.getLoc(), llvmDstTy,
                                                      diff, typeSizeVal);
    else
      resultVal = rewriter.create<mlir::LLVM::SDivOp>(op.getLoc(), llvmDstTy,
                                                      diff, typeSizeVal);
  }
  rewriter.replaceOp(op, resultVal);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMExpectOpLowering::matchAndRewrite(
    cir::ExpectOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::optional<llvm::APFloat> prob = op.getProb();
  if (!prob)
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExpectOp>(op, adaptor.getVal(),
                                                      adaptor.getExpected());
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExpectWithProbabilityOp>(
        op, adaptor.getVal(), adaptor.getExpected(), prob.value());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMVTableAddrPointOpLowering::matchAndRewrite(
    cir::VTableAddrPointOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  const auto *converter = getTypeConverter();
  auto targetType = converter->convertType(op.getType());
  mlir::Value symAddr = op.getSymAddr();
  llvm::SmallVector<mlir::LLVM::GEPArg> offsets;
  mlir::Type eltType;
  if (!symAddr) {
    symAddr = getValueForVTableSymbol(op, rewriter, getTypeConverter(),
                                      op.getNameAttr(), eltType);
    offsets = llvm::SmallVector<mlir::LLVM::GEPArg>{0, op.getVtableIndex(),
                                                    op.getAddressPointIndex()};
  } else {
    // Get indirect vtable address point retrieval
    symAddr = adaptor.getSymAddr();
    eltType = converter->convertType(symAddr.getType());
    offsets = llvm::SmallVector<mlir::LLVM::GEPArg>{op.getAddressPointIndex()};
  }

  assert(eltType && "Shouldn't ever be missing an eltType here");
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, targetType, eltType,
                                                 symAddr, offsets, true);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMStackSaveOpLowering::matchAndRewrite(
    cir::StackSaveOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto ptrTy = getTypeConverter()->convertType(op.getType());
  rewriter.replaceOpWithNewOp<mlir::LLVM::StackSaveOp>(op, ptrTy);
  return mlir::success();
}

#define GET_BUILTIN_LOWERING_CLASSES_DEF
#include "clang/CIR/Dialect/IR/CIRBuiltinsLowering.inc"
#undef GET_BUILTIN_LOWERING_CLASSES_DEF

mlir::LogicalResult CIRToLLVMUnreachableOpLowering::matchAndRewrite(
    cir::UnreachableOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMTrapOpLowering::matchAndRewrite(
    cir::TrapOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto loc = op->getLoc();
  rewriter.eraseOp(op);

  rewriter.create<mlir::LLVM::Trap>(loc);

  // Note that the call to llvm.trap is not a terminator in LLVM dialect.
  // So we must emit an additional llvm.unreachable to terminate the current
  // block.
  rewriter.create<mlir::LLVM::UnreachableOp>(loc);

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMInlineAsmOpLowering::matchAndRewrite(
    cir::InlineAsmOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type llResTy;
  if (op.getNumResults())
    llResTy = getTypeConverter()->convertType(op.getType(0));

  auto dialect = op.getAsmFlavor();
  auto llDialect = dialect == cir::AsmFlavor::x86_att
                       ? mlir::LLVM::AsmDialect::AD_ATT
                       : mlir::LLVM::AsmDialect::AD_Intel;

  std::vector<mlir::Attribute> opAttrs;
  auto llvmAttrName = mlir::LLVM::InlineAsmOp::getElementTypeAttrName();

  // this is for the lowering to LLVM from LLVm dialect. Otherwise, if we
  // don't have the result (i.e. void type as a result of operation), the
  // element type attribute will be attached to the whole instruction, but not
  // to the operand
  if (!op.getNumResults())
    opAttrs.push_back(mlir::Attribute());

  llvm::SmallVector<mlir::Value> llvmOperands;
  llvm::SmallVector<mlir::Value> cirOperands;
  for (size_t i = 0; i < op.getOperands().size(); ++i) {
    auto llvmOps = adaptor.getOperands()[i];
    auto cirOps = op.getOperands()[i];
    llvmOperands.insert(llvmOperands.end(), llvmOps.begin(), llvmOps.end());
    cirOperands.insert(cirOperands.end(), cirOps.begin(), cirOps.end());
  }

  // so far we infer the llvm dialect element type attr from
  // CIR operand type.
  for (std::size_t i = 0; i < op.getOperandAttrs().size(); ++i) {
    if (!op.getOperandAttrs()[i]) {
      opAttrs.push_back(mlir::Attribute());
      continue;
    }

    std::vector<mlir::NamedAttribute> attrs;
    auto typ = cast<cir::PointerType>(cirOperands[i].getType());
    auto typAttr =
        mlir::TypeAttr::get(getTypeConverter()->convertType(typ.getPointee()));

    attrs.push_back(rewriter.getNamedAttr(llvmAttrName, typAttr));
    auto newDict = rewriter.getDictionaryAttr(attrs);
    opAttrs.push_back(newDict);
  }

  rewriter.replaceOpWithNewOp<mlir::LLVM::InlineAsmOp>(
      op, llResTy, llvmOperands, op.getAsmStringAttr(), op.getConstraintsAttr(),
      op.getSideEffectsAttr(),
      /*is_align_stack*/ mlir::UnitAttr(),
      mlir::LLVM::AsmDialectAttr::get(getContext(), llDialect),
      rewriter.getArrayAttr(opAttrs));

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMPrefetchOpLowering::matchAndRewrite(
    cir::PrefetchOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::LLVM::Prefetch>(
      op, adaptor.getAddr(), adaptor.getIsWrite(), adaptor.getLocality(),
      /*DataCache*/ 1);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMSetBitfieldOpLowering::matchAndRewrite(
    cir::SetBitfieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  auto info = op.getBitfieldInfo();
  auto size = info.getSize();
  auto offset = info.getOffset();
  auto storageType = info.getStorageType();
  auto context = storageType.getContext();

  unsigned storageSize = 0;

  if (auto arTy = mlir::dyn_cast<cir::ArrayType>(storageType))
    storageSize = arTy.getSize() * 8;
  else if (auto intTy = mlir::dyn_cast<cir::IntType>(storageType))
    storageSize = intTy.getWidth();
  else
    llvm_unreachable(
        "Either ArrayType or IntType expected for bitfields storage");

  auto intType = mlir::IntegerType::get(context, storageSize);
  auto srcVal = createIntCast(rewriter, adaptor.getSrc(), intType);
  auto srcWidth = storageSize;
  auto resultVal = srcVal;

  if (storageSize != size) {
    assert(storageSize > size && "Invalid bitfield size.");

    mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), intType, adaptor.getAddr(), /* alignment */ 0,
        op.getIsVolatile());

    srcVal =
        createAnd(rewriter, srcVal, llvm::APInt::getLowBitsSet(srcWidth, size));
    resultVal = srcVal;
    srcVal = createShL(rewriter, srcVal, offset);

    // Mask out the original value.
    val = createAnd(rewriter, val,
                    ~llvm::APInt::getBitsSet(srcWidth, offset, offset + size));

    // Or together the unchanged values and the source value.
    srcVal = rewriter.create<mlir::LLVM::OrOp>(op.getLoc(), val, srcVal);
  }

  rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), srcVal, adaptor.getAddr(),
                                       /* alignment */ 0, op.getIsVolatile());

  auto resultTy = getTypeConverter()->convertType(op.getType());

  resultVal = createIntCast(rewriter, resultVal,
                            mlir::cast<mlir::IntegerType>(resultTy));

  if (info.getIsSigned()) {
    assert(size <= storageSize);
    unsigned highBits = storageSize - size;

    if (highBits) {
      resultVal = createShL(rewriter, resultVal, highBits);
      resultVal = createAShR(rewriter, resultVal, highBits);
    }
  }

  rewriter.replaceOp(op, resultVal);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMGetBitfieldOpLowering::matchAndRewrite(
    cir::GetBitfieldOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  auto info = op.getBitfieldInfo();
  auto size = info.getSize();
  auto offset = info.getOffset();
  auto storageType = info.getStorageType();
  auto context = storageType.getContext();
  unsigned storageSize = 0;

  if (auto arTy = mlir::dyn_cast<cir::ArrayType>(storageType))
    storageSize = arTy.getSize() * 8;
  else if (auto intTy = mlir::dyn_cast<cir::IntType>(storageType))
    storageSize = intTy.getWidth();
  else
    llvm_unreachable(
        "Either ArrayType or IntType expected for bitfields storage");

  auto intType = mlir::IntegerType::get(context, storageSize);

  mlir::Value val = rewriter.create<mlir::LLVM::LoadOp>(
      op.getLoc(), intType, adaptor.getAddr(), 0, op.getIsVolatile());
  val = rewriter.create<mlir::LLVM::BitcastOp>(op.getLoc(), intType, val);

  if (info.getIsSigned()) {
    assert(static_cast<unsigned>(offset + size) <= storageSize);
    unsigned highBits = storageSize - offset - size;
    val = createShL(rewriter, val, highBits);
    val = createAShR(rewriter, val, offset + highBits);
  } else {
    val = createLShR(rewriter, val, offset);

    if (static_cast<unsigned>(offset) + size < storageSize)
      val = createAnd(rewriter, val,
                      llvm::APInt::getLowBitsSet(storageSize, size));
  }

  auto resTy = getTypeConverter()->convertType(op.getType());
  auto newOp = createIntCast(
      rewriter, val, mlir::cast<mlir::IntegerType>(resTy), info.getIsSigned());
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMIsConstantOpLowering::matchAndRewrite(
    cir::IsConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // FIXME(cir): llvm.intr.is.constant returns i1 value but the LLVM Lowering
  // expects that cir.bool type will be lowered as i8 type.
  // So we have to insert zext here.
  auto isConstantOP =
      rewriter.create<mlir::LLVM::IsConstantOp>(op.getLoc(), adaptor.getVal());
  rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(op, rewriter.getI8Type(),
                                                  isConstantOP);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCmpThreeWayOpLowering::matchAndRewrite(
    cir::CmpThreeWayOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!op.isIntegralComparison() || !op.isStrongOrdering()) {
    op.emitError() << "unsupported three-way comparison type";
    return mlir::failure();
  }

  auto cmpInfo = op.getInfo();
  assert(cmpInfo.getLt() == -1 && cmpInfo.getEq() == 0 && cmpInfo.getGt() == 1);

  auto operandTy = mlir::cast<cir::IntType>(op.getLhs().getType());
  auto resultTy = op.getType();
  auto llvmIntrinsicName = getLLVMIntrinsicName(
      operandTy.isSigned(), operandTy.getWidth(), resultTy.getWidth());

  rewriter.setInsertionPoint(op);

  auto llvmLhs = adaptor.getLhs();
  auto llvmRhs = adaptor.getRhs();
  auto llvmResultTy = getTypeConverter()->convertType(resultTy);
  auto callIntrinsicOp =
      createCallLLVMIntrinsicOp(rewriter, op.getLoc(), llvmIntrinsicName,
                                llvmResultTy, {llvmLhs, llvmRhs});

  rewriter.replaceOp(op, callIntrinsicOp);
  return mlir::success();
}

std::string CIRToLLVMCmpThreeWayOpLowering::getLLVMIntrinsicName(
    bool signedCmp, unsigned operandWidth, unsigned resultWidth) {
  // The intrinsic's name takes the form:
  // `llvm.<scmp|ucmp>.i<resultWidth>.i<operandWidth>`

  std::string result = "llvm.";

  if (signedCmp)
    result.append("scmp.");
  else
    result.append("ucmp.");

  // Result type part.
  result.push_back('i');
  result.append(std::to_string(resultWidth));
  result.push_back('.');

  // Operand type part.
  result.push_back('i');
  result.append(std::to_string(operandWidth));

  return result;
}

mlir::LogicalResult CIRToLLVMReturnAddrOpLowering::matchAndRewrite(
    cir::ReturnAddrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.returnaddress",
                                   llvmPtrTy, adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMFrameAddrOpLowering::matchAndRewrite(
    cir::FrameAddrOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  replaceOpWithCallLLVMIntrinsicOp(rewriter, op, "llvm.frameaddress", llvmPtrTy,
                                   adaptor.getOperands());
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMClearCacheOpLowering::matchAndRewrite(
    cir::ClearCacheOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto begin = adaptor.getBegin();
  auto end = adaptor.getEnd();
  auto intrinNameAttr =
      mlir::StringAttr::get(op.getContext(), "llvm.clear_cache");
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallIntrinsicOp>(
      op, mlir::Type{}, intrinNameAttr, mlir::ValueRange{begin, end});

  return mlir::success();
}

mlir::LogicalResult CIRToLLVMEhTypeIdOpLowering::matchAndRewrite(
    cir::EhTypeIdOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value addrOp = rewriter.create<mlir::LLVM::AddressOfOp>(
      op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
      op.getTypeSymAttr());
  mlir::LLVM::CallIntrinsicOp newOp = createCallLLVMIntrinsicOp(
      rewriter, op.getLoc(), "llvm.eh.typeid.for.p0", rewriter.getI32Type(),
      mlir::ValueRange{addrOp});
  rewriter.replaceOp(op, newOp);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMCatchParamOpLowering::matchAndRewrite(
    cir::CatchParamOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (op.isBegin()) {
    // Get or create `declare ptr @__cxa_begin_catch(ptr)`
    StringRef fnName = "__cxa_begin_catch";
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {llvmPtrTy},
                                                  /*isVarArg=*/false);
    getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, mlir::TypeRange{llvmPtrTy}, fnName,
        mlir::ValueRange{adaptor.getExceptionPtr()});
    return mlir::success();
  } else if (op.isEnd()) {
    StringRef fnName = "__cxa_end_catch";
    auto fnTy = mlir::LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {},
        /*isVarArg=*/false);
    getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
    rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), mlir::TypeRange{}, fnName,
                                        mlir::ValueRange{});
    rewriter.eraseOp(op);
    return mlir::success();
  }
  llvm_unreachable("only begin/end supposed to make to lowering stage");
  return mlir::failure();
}

mlir::LogicalResult CIRToLLVMResumeOpLowering::matchAndRewrite(
    cir::ResumeOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // %lpad.val = insertvalue { ptr, i32 } poison, ptr %exception_ptr, 0
  // %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %selector, 1
  // resume { ptr, i32 } %lpad.val2
  SmallVector<int64_t> slotIdx = {0};
  SmallVector<int64_t> selectorIdx = {1};
  auto llvmLandingPadStructTy = getLLVMLandingPadStructTy(rewriter);
  mlir::Value poison = rewriter.create<mlir::LLVM::PoisonOp>(
      op.getLoc(), llvmLandingPadStructTy);

  mlir::Value slot = rewriter.create<mlir::LLVM::InsertValueOp>(
      op.getLoc(), poison, adaptor.getExceptionPtr(), slotIdx);
  mlir::Value selector = rewriter.create<mlir::LLVM::InsertValueOp>(
      op.getLoc(), slot, adaptor.getTypeId(), selectorIdx);

  rewriter.replaceOpWithNewOp<mlir::LLVM::ResumeOp>(op, selector);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAllocExceptionOpLowering::matchAndRewrite(
    cir::AllocExceptionOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Get or create `declare ptr @__cxa_allocate_exception(i64)`
  StringRef fnName = "__cxa_allocate_exception";
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto int64Ty = mlir::IntegerType::get(rewriter.getContext(), 64);
  auto fnTy = mlir::LLVM::LLVMFunctionType::get(llvmPtrTy, {int64Ty},
                                                /*isVarArg=*/false);
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  auto size = rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(),
                                                      adaptor.getSizeAttr());
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{llvmPtrTy}, fnName, mlir::ValueRange{size});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMFreeExceptionOpLowering::matchAndRewrite(
    cir::FreeExceptionOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Get or create `declare void @__cxa_free_exception(ptr)`
  StringRef fnName = "__cxa_free_exception";
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto voidTy = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
  auto fnTy = mlir::LLVM::LLVMFunctionType::get(voidTy, {llvmPtrTy},
                                                /*isVarArg=*/false);
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{}, fnName, mlir::ValueRange{adaptor.getPtr()});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMThrowOpLowering::matchAndRewrite(
    cir::ThrowOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Get or create `declare void @__cxa_throw(ptr, ptr, ptr)`
  StringRef fnName = "__cxa_throw";
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto voidTy = mlir::LLVM::LLVMVoidType::get(rewriter.getContext());
  auto fnTy = mlir::LLVM::LLVMFunctionType::get(
      voidTy, {llvmPtrTy, llvmPtrTy, llvmPtrTy},
      /*isVarArg=*/false);
  getOrCreateLLVMFuncOp(rewriter, op, fnName, fnTy);
  mlir::Value typeInfo = rewriter.create<mlir::LLVM::AddressOfOp>(
      op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
      adaptor.getTypeInfoAttr());

  mlir::Value dtor;
  if (op.getDtor()) {
    dtor = rewriter.create<mlir::LLVM::AddressOfOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        adaptor.getDtorAttr());
  } else {
    dtor = rewriter.create<mlir::LLVM::ZeroOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));
  }
  rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
      op, mlir::TypeRange{}, fnName,
      mlir::ValueRange{adaptor.getExceptionPtr(), typeInfo, dtor});
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMIsFPClassOpLowering::matchAndRewrite(
    cir::IsFPClassOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto src = adaptor.getSrc();
  auto flags = adaptor.getFlags();
  auto retTy = rewriter.getI1Type();

  auto loc = op->getLoc();

  auto intrinsic =
      rewriter.create<mlir::LLVM::IsFPClass>(loc, retTy, src, flags);
  // FIMXE: CIR now will convert cir::BoolType to i8 type unconditionally.
  // Remove this conversion after we fix
  // https://github.com/llvm/clangir/issues/480
  auto converted = rewriter.create<mlir::LLVM::ZExtOp>(
      loc, rewriter.getI8Type(), intrinsic->getResult(0));

  rewriter.replaceOp(op, converted);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMAbsOpLowering::matchAndRewrite(
    cir::AbsOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto resTy = this->getTypeConverter()->convertType(op.getType());
  auto absOp = rewriter.create<mlir::LLVM::AbsOp>(
      op.getLoc(), resTy, adaptor.getOperands()[0], adaptor.getPoison());
  rewriter.replaceOp(op, absOp);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMPtrMaskOpLowering::matchAndRewrite(
    cir::PtrMaskOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // FIXME: We'd better to lower to mlir::LLVM::PtrMaskOp if it exists.
  // So we have to make it manually here by following:
  // https://llvm.org/docs/LangRef.html#llvm-ptrmask-intrinsic
  auto loc = op.getLoc();
  auto mask = op.getMask();

  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  mlir::DataLayout layout(moduleOp);
  auto iPtrIdxValue = layout.getTypeSizeInBits(mask.getType());
  auto iPtrIdx = mlir::IntegerType::get(moduleOp->getContext(), iPtrIdxValue);

  auto intPtr = rewriter.create<mlir::LLVM::PtrToIntOp>(
      loc, iPtrIdx, adaptor.getPtr()); // this may truncate
  mlir::Value masked =
      rewriter.create<mlir::LLVM::AndOp>(loc, intPtr, adaptor.getMask());
  mlir::Value diff = rewriter.create<mlir::LLVM::SubOp>(loc, intPtr, masked);
  rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
      op, getTypeConverter()->convertType(op.getType()),
      mlir::IntegerType::get(moduleOp->getContext(), 8), adaptor.getPtr(),
      diff);
  return mlir::success();
}

mlir::LogicalResult CIRToLLVMSignBitOpLowering::matchAndRewrite(
    cir::SignBitOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  assert(!::cir::MissingFeatures::isPPC_FP128Ty());

  mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
  int width = layout.getTypeSizeInBits(op.getInput().getType());
  if (auto longDoubleType =
          mlir::dyn_cast<cir::LongDoubleType>(op.getInput().getType())) {
    if (mlir::isa<cir::FP80Type>(longDoubleType.getUnderlying())) {
      // If the underlying type of LongDouble is FP80Type,
      // DataLayout::getTypeSizeInBits returns 128.
      // See https://github.com/llvm/clangir/issues/1057.
      // Set the width to 80 manually.
      width = 80;
    }
  }
  auto intTy = mlir::IntegerType::get(rewriter.getContext(), width);
  auto bitcast = rewriter.create<mlir::LLVM::BitcastOp>(op->getLoc(), intTy,
                                                        adaptor.getInput());
  auto zero = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), intTy, 0);
  auto cmpResult = rewriter.create<mlir::LLVM::ICmpOp>(
      op.getLoc(), mlir::LLVM::ICmpPredicate::slt, bitcast.getResult(), zero);
  auto converted = rewriter.create<mlir::LLVM::ZExtOp>(
      op.getLoc(), getTypeConverter()->convertType(op.getType()), cmpResult);
  rewriter.replaceOp(op, converted);
  return mlir::success();
}

void populateCIRToLLVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &converter,
    mlir::DataLayout &dataLayout, cir::LowerModule *lowerModule,
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap) {
  patterns.add<CIRToLLVMReturnOpLowering>(patterns.getContext());
  patterns.add<CIRToLLVMAllocaOpLowering>(converter, dataLayout,
                                          stringGlobalsMap, argStringGlobalsMap,
                                          argsVarMap, patterns.getContext());
  patterns.add<
      // clang-format off
      CIRToLLVMBaseDataMemberOpLowering,
      CIRToLLVMConstantOpLowering,
      CIRToLLVMDerivedDataMemberOpLowering,
      CIRToLLVMGetRuntimeMemberOpLowering,
      CIRToLLVMGlobalOpLowering
      // clang-format on
      >(converter, patterns.getContext(), lowerModule);
  patterns.add<
      // clang-format off
      CIRToLLVMAbsOpLowering,
      CIRToLLVMAllocExceptionOpLowering,
      CIRToLLVMAssumeAlignedOpLowering,
      CIRToLLVMAssumeOpLowering,
      CIRToLLVMAssumeSepStorageOpLowering,
      CIRToLLVMAtomicCmpXchgLowering,
      CIRToLLVMAtomicFetchLowering,
      CIRToLLVMAtomicXchgLowering,
      CIRToLLVMBaseClassAddrOpLowering,
      CIRToLLVMBinOpLowering,
      CIRToLLVMBinOpOverflowOpLowering,
      CIRToLLVMBitClrsbOpLowering,
      CIRToLLVMBitClzOpLowering,
      CIRToLLVMBitCtzOpLowering,
      CIRToLLVMBitFfsOpLowering,
      CIRToLLVMBitParityOpLowering,
      CIRToLLVMBitPopcountOpLowering,
      CIRToLLVMBrCondOpLowering,
      CIRToLLVMBrOpLowering,
      CIRToLLVMByteswapOpLowering,
      CIRToLLVMCallOpLowering,
      CIRToLLVMCastOpLowering,
      CIRToLLVMCatchParamOpLowering,
      CIRToLLVMClearCacheOpLowering,
      CIRToLLVMCmpOpLowering,
      CIRToLLVMCmpThreeWayOpLowering,
      CIRToLLVMComplexCreateOpLowering,
      CIRToLLVMComplexImagOpLowering,
      CIRToLLVMComplexImagPtrOpLowering,
      CIRToLLVMComplexRealOpLowering,
      CIRToLLVMComplexRealPtrOpLowering,
      CIRToLLVMCopyOpLowering,
      CIRToLLVMDerivedClassAddrOpLowering,
      CIRToLLVMEhInflightOpLowering,
      CIRToLLVMEhTypeIdOpLowering,
      CIRToLLVMExpectOpLowering,
      CIRToLLVMFrameAddrOpLowering,
      CIRToLLVMFreeExceptionOpLowering,
      CIRToLLVMFuncOpLowering,
      CIRToLLVMGetBitfieldOpLowering,
      CIRToLLVMGetGlobalOpLowering,
      CIRToLLVMGetMemberOpLowering,
      CIRToLLVMInlineAsmOpLowering,
      CIRToLLVMIsConstantOpLowering,
      CIRToLLVMIsFPClassOpLowering,
      CIRToLLVMLLVMIntrinsicCallOpLowering,
      CIRToLLVMLoadOpLowering,
      CIRToLLVMMemChrOpLowering,
      CIRToLLVMMemCpyInlineOpLowering,
      CIRToLLVMMemCpyOpLowering,
      CIRToLLVMMemMoveOpLowering,
      CIRToLLVMMemSetInlineOpLowering,
      CIRToLLVMMemSetOpLowering,
      CIRToLLVMObjSizeOpLowering,
      CIRToLLVMPrefetchOpLowering,
      CIRToLLVMPtrDiffOpLowering,
      CIRToLLVMPtrMaskOpLowering,
      CIRToLLVMPtrStrideOpLowering,
      CIRToLLVMResumeOpLowering,
      CIRToLLVMReturnAddrOpLowering,
      CIRToLLVMRotateOpLowering,
      CIRToLLVMSelectOpLowering,
      CIRToLLVMSetBitfieldOpLowering,
      CIRToLLVMShiftOpLowering,
      CIRToLLVMSignBitOpLowering,
      CIRToLLVMStackSaveOpLowering,
      CIRToLLVMStoreOpLowering,
      CIRToLLVMSwitchFlatOpLowering,
      CIRToLLVMThrowOpLowering,
      CIRToLLVMTrapOpLowering,
      CIRToLLVMTryCallOpLowering,
      CIRToLLVMUnaryOpLowering,
      CIRToLLVMUnreachableOpLowering,
      CIRToLLVMVAArgOpLowering,
      CIRToLLVMVACopyOpLowering,
      CIRToLLVMVAEndOpLowering,
      CIRToLLVMVAStartOpLowering,
      CIRToLLVMVecCmpOpLowering,
      CIRToLLVMVecCreateOpLowering,
      CIRToLLVMVecShuffleDynamicOpLowering,
      CIRToLLVMVecShuffleOpLowering,
      CIRToLLVMVecSplatOpLowering,
      CIRToLLVMVecTernaryOpLowering,
      CIRToLLVMVTableAddrPointOpLowering,
      CIRToLLVMVTTAddrPointOpLowering
#define GET_BUILTIN_LOWERING_LIST
#include "clang/CIR/Dialect/IR/CIRBuiltinsLowering.inc"
#undef GET_BUILTIN_LOWERING_LIST
      // clang-format on
      >(converter, patterns.getContext());
}

std::unique_ptr<cir::LowerModule> prepareLowerModule(mlir::ModuleOp module) {
  mlir::PatternRewriter rewriter{module->getContext()};
  // If the triple is not present, e.g. CIR modules parsed from text, we
  // cannot init LowerModule properly.
  assert(!cir::MissingFeatures::makeTripleAlwaysPresent());
  if (!module->hasAttr(cir::CIRDialect::getTripleAttrName()))
    return {};
  return cir::createLowerModule(module, rewriter);
}

// FIXME: change the type of lowerModule to `LowerModule &` to have better
// lambda capturing experience. Also blocked by makeTripleAlwaysPresent.
void prepareTypeConverter(mlir::LLVMTypeConverter &converter,
                          mlir::DataLayout &dataLayout,
                          cir::LowerModule *lowerModule) {
  converter.addConversion(
      [&, lowerModule](cir::PointerType type) -> mlir::Type {
        // Drop pointee type since LLVM dialect only allows opaque pointers.

        auto addrSpace =
            mlir::cast_if_present<cir::AddressSpaceAttr>(type.getAddrSpace());
        // Null addrspace attribute indicates the default addrspace.
        if (!addrSpace)
          return mlir::LLVM::LLVMPointerType::get(type.getContext());

        assert(lowerModule && "CIR AS map is not available");
        // Pass through target addrspace and map CIR addrspace to LLVM addrspace
        // by querying the target info.
        unsigned targetAS =
            addrSpace.isTarget()
                ? addrSpace.getTargetValue()
                : lowerModule->getTargetLoweringInfo()
                      .getTargetAddrSpaceFromCIRAddrSpace(addrSpace);

        return mlir::LLVM::LLVMPointerType::get(type.getContext(), targetAS);
      });
  converter.addConversion(
      [&, lowerModule](cir::DataMemberType type) -> mlir::Type {
        assert(lowerModule && "CXXABI is not available");
        mlir::Type abiType =
            lowerModule->getCXXABI().lowerDataMemberType(type, converter);
        return converter.convertType(abiType);
      });
  converter.addConversion([&](cir::ArrayType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](cir::VectorType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::LLVM::getFixedVectorType(ty, type.getSize());
  });
  converter.addConversion([&](cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8,
                                  mlir::IntegerType::Signless);
  });
  converter.addConversion([&](cir::IntType type) -> mlir::Type {
    // LLVM doesn't work with signed types, so we drop the CIR signs here.
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
  converter.addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::FloatType::getF32(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::FloatType::getF64(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::FloatType::getF80(type.getContext());
  });
  converter.addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::FloatType::getF128(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::FloatType::getF16(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::FloatType::getBF16(type.getContext());
  });
  converter.addConversion([&](cir::ComplexType type) -> mlir::Type {
    // A complex type is lowered to an LLVM struct that contains the real and
    // imaginary part as data fields.
    mlir::Type elementTy = converter.convertType(type.getElementTy());
    mlir::Type structFields[2] = {elementTy, elementTy};
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                  structFields);
  });
  converter.addConversion([&](cir::FuncType type) -> mlir::Type {
    auto result = converter.convertType(type.getReturnType());
    llvm::SmallVector<mlir::Type> arguments;
    if (converter.convertTypes(type.getInputs(), arguments).failed())
      llvm_unreachable("Failed to convert function type parameters");
    auto varArg = type.isVarArg();
    return mlir::LLVM::LLVMFunctionType::get(result, arguments, varArg);
  });
  converter.addConversion([&](cir::StructType type) -> mlir::Type {
    // FIXME(cir): create separate unions, struct, and classes types.
    // Convert struct members.
    llvm::SmallVector<mlir::Type> llvmMembers;
    switch (type.getKind()) {
    case cir::StructType::Class:
      // TODO(cir): This should be properly validated.
    case cir::StructType::Struct:
      for (auto ty : type.getMembers())
        llvmMembers.push_back(converter.convertType(ty));
      break;
    // Unions are lowered as only the largest member.
    case cir::StructType::Union: {
      auto largestMember = type.getLargestMember(dataLayout);
      if (largestMember)
        llvmMembers.push_back(converter.convertType(largestMember));
      break;
    }
    }

    // Struct has a name: lower as an identified struct.
    mlir::LLVM::LLVMStructType llvmStruct;
    if (type.getName()) {
      llvmStruct = mlir::LLVM::LLVMStructType::getIdentified(
          type.getContext(), type.getPrefixedName());
      if (llvmStruct.setBody(llvmMembers, /*isPacked=*/type.getPacked())
              .failed())
        llvm_unreachable("Failed to set body of struct");
    } else { // Struct has no name: lower as literal struct.
      llvmStruct = mlir::LLVM::LLVMStructType::getLiteral(
          type.getContext(), llvmMembers, /*isPacked=*/type.getPacked());
    }

    return llvmStruct;
  });
  converter.addConversion([&](cir::VoidType type) -> mlir::Type {
    return mlir::LLVM::LLVMVoidType::get(type.getContext());
  });
}

void buildCtorDtorList(
    mlir::ModuleOp module, StringRef globalXtorName, StringRef llvmXtorName,
    llvm::function_ref<std::pair<StringRef, int>(mlir::Attribute)> createXtor) {
  llvm::SmallVector<std::pair<StringRef, int>, 2> globalXtors;
  for (auto namedAttr : module->getAttrs()) {
    if (namedAttr.getName() == globalXtorName) {
      for (auto attr : mlir::cast<mlir::ArrayAttr>(namedAttr.getValue()))
        globalXtors.emplace_back(createXtor(attr));
      break;
    }
  }

  if (globalXtors.empty())
    return;

  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Create a global array llvm.global_ctors with element type of
  // struct { i32, ptr, ptr }
  auto CtorPFTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  llvm::SmallVector<mlir::Type> CtorStructFields;
  CtorStructFields.push_back(builder.getI32Type());
  CtorStructFields.push_back(CtorPFTy);
  CtorStructFields.push_back(CtorPFTy);

  auto CtorStructTy = mlir::LLVM::LLVMStructType::getLiteral(
      builder.getContext(), CtorStructFields);
  auto CtorStructArrayTy =
      mlir::LLVM::LLVMArrayType::get(CtorStructTy, globalXtors.size());

  auto loc = module.getLoc();
  auto newGlobalOp = builder.create<mlir::LLVM::GlobalOp>(
      loc, CtorStructArrayTy, true, mlir::LLVM::Linkage::Appending,
      llvmXtorName, mlir::Attribute());

  newGlobalOp.getRegion().push_back(new mlir::Block());
  builder.setInsertionPointToEnd(newGlobalOp.getInitializerBlock());

  mlir::Value result =
      builder.create<mlir::LLVM::UndefOp>(loc, CtorStructArrayTy);

  for (uint64_t I = 0; I < globalXtors.size(); I++) {
    auto fn = globalXtors[I];
    mlir::Value structInit =
        builder.create<mlir::LLVM::UndefOp>(loc, CtorStructTy);
    mlir::Value initPriority = builder.create<mlir::LLVM::ConstantOp>(
        loc, CtorStructFields[0], fn.second);
    mlir::Value initFuncAddr = builder.create<mlir::LLVM::AddressOfOp>(
        loc, CtorStructFields[1], fn.first);
    mlir::Value initAssociate =
        builder.create<mlir::LLVM::ZeroOp>(loc, CtorStructFields[2]);
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initPriority, 0);
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initFuncAddr, 1);
    // TODO: handle associated data for initializers.
    structInit = builder.create<mlir::LLVM::InsertValueOp>(loc, structInit,
                                                           initAssociate, 2);
    result =
        builder.create<mlir::LLVM::InsertValueOp>(loc, result, structInit, I);
  }

  builder.create<mlir::LLVM::ReturnOp>(loc, result);
}

// The unreachable code is not lowered by applyPartialConversion function
// since it traverses blocks in the dominance order. At the same time we
// do need to lower such code - otherwise verification errors occur.
// For instance, the next CIR code:
//
//    cir.func @foo(%arg0: !s32i) -> !s32i {
//      %4 = cir.cast(int_to_bool, %arg0 : !s32i), !cir.bool
//      cir.if %4 {
//        %5 = cir.const #cir.int<1> : !s32i
//        cir.return %5 : !s32i
//      } else {
//        %5 = cir.const #cir.int<0> : !s32i
//       cir.return %5 : !s32i
//      }
//     cir.return %arg0 : !s32i
//    }
//
// contains an unreachable return operation (the last one). After the flattening
// pass it will be placed into the unreachable block. And the possible error
// after the lowering pass is: error: 'cir.return' op expects parent op to be
// one of 'cir.func, cir.scope, cir.if ... The reason that this operation was
// not lowered and the new parent is llvm.func.
//
// In the future we may want to get rid of this function and use DCE pass or
// something similar. But now we need to guarantee the absence of the dialect
// verification errors.
void collect_unreachable(mlir::Operation *parent,
                         llvm::SmallVector<mlir::Operation *> &ops) {

  llvm::SmallVector<mlir::Block *> unreachable_blocks;
  parent->walk([&](mlir::Block *blk) { // check
    if (blk->hasNoPredecessors() && !blk->isEntryBlock())
      unreachable_blocks.push_back(blk);
  });

  std::set<mlir::Block *> visited;
  for (auto *root : unreachable_blocks) {
    // We create a work list for each unreachable block.
    // Thus we traverse operations in some order.
    std::deque<mlir::Block *> workList;
    workList.push_back(root);

    while (!workList.empty()) {
      auto *blk = workList.back();
      workList.pop_back();
      if (visited.count(blk))
        continue;
      visited.emplace(blk);

      for (auto &op : *blk)
        ops.push_back(&op);

      for (auto it = blk->succ_begin(); it != blk->succ_end(); ++it)
        workList.push_back(*it);
    }
  }
}

void ConvertCIRToLLVMPass::buildGlobalAnnotationsVar(
    llvm::StringMap<mlir::LLVM::GlobalOp> &stringGlobalsMap,
    llvm::StringMap<mlir::LLVM::GlobalOp> &argStringGlobalsMap,
    llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> &argsVarMap) {
  mlir::ModuleOp module = getOperation();
  mlir::Attribute attr =
      module->getAttr(cir::CIRDialect::getGlobalAnnotationsAttrName());
  if (!attr)
    return;
  if (auto globalAnnotValues =
          mlir::dyn_cast<cir::GlobalAnnotationValuesAttr>(attr)) {
    auto annotationValuesArray =
        mlir::dyn_cast<mlir::ArrayAttr>(globalAnnotValues.getAnnotations());
    if (!annotationValuesArray || annotationValuesArray.empty())
      return;
    mlir::OpBuilder globalVarBuilder(module.getContext());
    globalVarBuilder.setInsertionPointToEnd(&module.getBodyRegion().front());

    // Create a global array for annotation values with element type of
    // struct { ptr, ptr, ptr, i32, ptr }
    mlir::LLVM::LLVMPointerType annoPtrTy =
        mlir::LLVM::LLVMPointerType::get(globalVarBuilder.getContext());
    llvm::SmallVector<mlir::Type> annoStructFields;
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(annoPtrTy);
    annoStructFields.push_back(globalVarBuilder.getI32Type());
    annoStructFields.push_back(annoPtrTy);

    mlir::LLVM::LLVMStructType annoStructTy =
        mlir::LLVM::LLVMStructType::getLiteral(globalVarBuilder.getContext(),
                                               annoStructFields);
    mlir::LLVM::LLVMArrayType annoStructArrayTy =
        mlir::LLVM::LLVMArrayType::get(annoStructTy,
                                       annotationValuesArray.size());
    mlir::Location moduleLoc = module.getLoc();
    auto annotationGlobalOp = globalVarBuilder.create<mlir::LLVM::GlobalOp>(
        moduleLoc, annoStructArrayTy, false, mlir::LLVM::Linkage::Appending,
        "llvm.global.annotations", mlir::Attribute());
    annotationGlobalOp.setSection(llvmMetadataSectionName);
    annotationGlobalOp.getRegion().push_back(new mlir::Block());
    mlir::OpBuilder varInitBuilder(module.getContext());
    varInitBuilder.setInsertionPointToEnd(
        annotationGlobalOp.getInitializerBlock());
    // Globals created for annotation strings and args to be
    // placed before the var llvm.global.annotations.
    // This is consistent with clang code gen.
    globalVarBuilder.setInsertionPoint(annotationGlobalOp);

    mlir::Value result = varInitBuilder.create<mlir::LLVM::UndefOp>(
        moduleLoc, annoStructArrayTy);

    int idx = 0;
    for (mlir::Attribute entry : annotationValuesArray) {
      auto annotValue = cast<mlir::ArrayAttr>(entry);
      mlir::Value valueEntry =
          varInitBuilder.create<mlir::LLVM::UndefOp>(moduleLoc, annoStructTy);
      SmallVector<mlir::Value, 4> vals;

      auto globalValueName = mlir::cast<mlir::StringAttr>(annotValue[0]);
      mlir::Operation *globalValue =
          mlir::SymbolTable::lookupSymbolIn(module, globalValueName);
      // The first field is ptr to the global value
      auto globalValueFld = varInitBuilder.create<mlir::LLVM::AddressOfOp>(
          moduleLoc, annoPtrTy, globalValueName);
      vals.push_back(globalValueFld->getResult(0));

      cir::AnnotationAttr annot =
          mlir::cast<cir::AnnotationAttr>(annotValue[1]);
      lowerAnnotationValue(moduleLoc, globalValue->getLoc(), annot, module,
                           varInitBuilder, globalVarBuilder, stringGlobalsMap,
                           argStringGlobalsMap, argsVarMap, vals);
      for (unsigned valIdx = 0, endIdx = vals.size(); valIdx != endIdx;
           ++valIdx) {
        valueEntry = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
            moduleLoc, valueEntry, vals[valIdx], valIdx);
      }
      result = varInitBuilder.create<mlir::LLVM::InsertValueOp>(
          moduleLoc, result, valueEntry, idx++);
    }
    varInitBuilder.create<mlir::LLVM::ReturnOp>(moduleLoc, result);
  }
}

void ConvertCIRToLLVMPass::processCIRAttrs(mlir::ModuleOp module) {
  // Lower the module attributes to LLVM equivalents.
  if (auto tripleAttr = module->getAttr(cir::CIRDialect::getTripleAttrName()))
    module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                    tripleAttr);

  // Strip the CIR attributes.
  module->removeAttr(cir::CIRDialect::getSOBAttrName());
  module->removeAttr(cir::CIRDialect::getLangAttrName());
  module->removeAttr(cir::CIRDialect::getTripleAttrName());
}

void ConvertCIRToLLVMPass::runOnOperation() {
  llvm::TimeTraceScope scope("Convert CIR to LLVM Pass");

  auto module = getOperation();
  mlir::DataLayout dataLayout(module);
  mlir::LLVMTypeConverter converter(&getContext());
  std::unique_ptr<cir::LowerModule> lowerModule = prepareLowerModule(module);
  prepareTypeConverter(converter, dataLayout, lowerModule.get());

  mlir::RewritePatternSet patterns(&getContext());

  // Track globals created for annotation related strings
  llvm::StringMap<mlir::LLVM::GlobalOp> stringGlobalsMap;
  // Track globals created for annotation arg related strings.
  // They are different from annotation strings, as strings used in args
  // are not in llvmMetadataSectionName, and also has aligment 1.
  llvm::StringMap<mlir::LLVM::GlobalOp> argStringGlobalsMap;
  // Track globals created for annotation args.
  llvm::MapVector<mlir::ArrayAttr, mlir::LLVM::GlobalOp> argsVarMap;

  populateCIRToLLVMConversionPatterns(patterns, converter, dataLayout,
                                      lowerModule.get(), stringGlobalsMap,
                                      argStringGlobalsMap, argsVarMap);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(getContext());
  using namespace cir;
  // clang-format off
  target.addLegalOp<mlir::ModuleOp
                    // ,AllocaOp
                    // ,BrCondOp
                    // ,BrOp
                    // ,CallOp
                    // ,CastOp
                    // ,CmpOp
                    // ,ConstantOp
                    // ,FuncOp
                    // ,LoadOp
                    // ,ReturnOp
                    // ,StoreOp
                    // ,YieldOp
                    >();
  // clang-format on
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::BuiltinDialect, cir::CIRDialect,
                           mlir::func::FuncDialect>();

  // Allow operations that will be lowered directly to LLVM IR.
  target.addLegalOp<mlir::LLVM::ZeroOp>();

  processCIRAttrs(module);

  llvm::SmallVector<mlir::Operation *> ops;
  ops.push_back(module);
  collect_unreachable(module, ops);

  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();

  // Emit the llvm.global_ctors array.
  buildCtorDtorList(module, cir::CIRDialect::getGlobalCtorsAttrName(),
                    "llvm.global_ctors", [](mlir::Attribute attr) {
                      assert(mlir::isa<cir::GlobalCtorAttr>(attr) &&
                             "must be a GlobalCtorAttr");
                      auto ctorAttr = mlir::cast<cir::GlobalCtorAttr>(attr);
                      return std::make_pair(ctorAttr.getName(),
                                            ctorAttr.getPriority());
                    });
  // Emit the llvm.global_dtors array.
  buildCtorDtorList(module, cir::CIRDialect::getGlobalDtorsAttrName(),
                    "llvm.global_dtors", [](mlir::Attribute attr) {
                      assert(mlir::isa<cir::GlobalDtorAttr>(attr) &&
                             "must be a GlobalDtorAttr");
                      auto dtorAttr = mlir::cast<cir::GlobalDtorAttr>(attr);
                      return std::make_pair(dtorAttr.getName(),
                                            dtorAttr.getPriority());
                    });
  buildGlobalAnnotationsVar(stringGlobalsMap, argStringGlobalsMap, argsVarMap);
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

struct TBAAPass
    : public mlir::PassWrapper<TBAAPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    llvm::TimeTraceScope scope("TBAA Pass");
    auto moduleOp = getOperation();
    moduleOp->walk([&](cir::LoadOp loadOp) { lowerTBAAAttr(loadOp); });
    moduleOp->walk([&](cir::StoreOp storeOp) { lowerTBAAAttr(storeOp); });
  }

private:
  template <typename Op> void lowerTBAAAttr(Op op) {
    auto tbaa = lowerTBAAToArrayAttr(op.getContext(), op.getTbaa());
    op.setTbaaAttr(tbaa);
  }

  mlir::LLVM::TBAATagAttr lowerTBAAAttr(mlir::MLIRContext *ctx,
                                        mlir::Attribute attr) {
    if (auto scalarType =
            mlir::dyn_cast<cir::TBAAScalarTypeDescriptorAttr>(attr)) {
      auto accessType = lowerScalarType(ctx, scalarType);
      return mlir::LLVM::TBAATagAttr::get(accessType, accessType, 0);
    }
    if (auto tbaaTagAttr = mlir::dyn_cast<cir::TBAATagAttr>(attr)) {
      assert(mlir::isa<cir::TBAAStructTypeDescriptorAttr>(
          tbaaTagAttr.getBaseType()));
      auto baseType = lowerStructType(
          ctx, mlir::dyn_cast<cir::TBAAStructTypeDescriptorAttr>(
                   tbaaTagAttr.getBaseType()));
      assert(mlir::isa<cir::TBAAScalarTypeDescriptorAttr>(
          tbaaTagAttr.getAccessType()));
      auto accessType = lowerScalarType(
          ctx, mlir::dyn_cast<cir::TBAAScalarTypeDescriptorAttr>(
                   tbaaTagAttr.getAccessType()));
      return mlir::LLVM::TBAATagAttr::get(baseType, accessType,
                                          tbaaTagAttr.getOffset());
    }
    llvm_unreachable("unknown tbaa type");
  }

  mlir::ArrayAttr lowerTBAAToArrayAttr(mlir::MLIRContext *ctx,
                                       std::optional<mlir::ArrayAttr> tbaa) {
    if (!tbaa) {
      return mlir::ArrayAttr();
    }
    auto tbaaTagArray = *tbaa;
    if (tbaaTagArray.empty()) {
      return mlir::ArrayAttr();
    }
    llvm::SmallVector<mlir::Attribute, 2> tags;
    for (auto attr : tbaaTagArray) {
      auto tag = lowerTBAAAttr(ctx, attr);
      tags.push_back(tag);
    }
    return mlir::ArrayAttr::get(ctx, tags);
  }

  mlir::LLVM::TBAATypeDescriptorAttr
  createScalarTypeNode(mlir::MLIRContext *ctx, llvm::StringRef typeName,
                       mlir::LLVM::TBAANodeAttr parent,
                       std::size_t offset) const {
    llvm::SmallVector<mlir::LLVM::TBAAMemberAttr, 2> intMember;
    intMember.push_back(mlir::LLVM::TBAAMemberAttr::get(ctx, parent, 0));

    return mlir::LLVM::TBAATypeDescriptorAttr::get(
        ctx, typeName, llvm::ArrayRef<mlir::LLVM::TBAAMemberAttr>(intMember));
  }

  mlir::LLVM::TBAARootAttr getRoot(mlir::MLIRContext *ctx) const {
    return mlir::LLVM::TBAARootAttr::get(
        ctx, mlir::StringAttr::get(ctx, "Simple C/C++ TBAA"));
  }

  mlir::LLVM::TBAATypeDescriptorAttr getChar(mlir::MLIRContext *ctx) const {
    return createScalarTypeNode(ctx, "omnipotent char", getRoot(ctx),
                                /* Size= */ 1);
  }

  mlir::LLVM::TBAATypeDescriptorAttr
  lowerScalarType(mlir::MLIRContext *ctx,
                  cir::TBAAScalarTypeDescriptorAttr scalarType) const {
    // special handle for omnipotent char
    if (scalarType.getId() == "omnipotent char") {
      return getChar(ctx);
    }
    return createScalarTypeNode(ctx, scalarType.getId(), getChar(ctx), 0);
  }
  mlir::LLVM::TBAATypeDescriptorAttr
  lowerStructType(mlir::MLIRContext *ctx,
                  cir::TBAAStructTypeDescriptorAttr structType) const {
    auto ret = structCache[structType];
    if (ret) {
      return ret;
    }
    llvm::SmallVector<mlir::LLVM::TBAAMemberAttr, 2> members;
    for (auto &member : structType.getMembers()) {
      auto desc = member.getTypeDesc();
      if (auto scalarType =
              mlir::dyn_cast<cir::TBAAScalarTypeDescriptorAttr>(desc)) {
        members.push_back(mlir::LLVM::TBAAMemberAttr::get(
            lowerScalarType(ctx, scalarType), member.getOffset()));
      } else if (auto subStructType =
                     mlir::dyn_cast<cir::TBAAStructTypeDescriptorAttr>(desc)) {
        auto subStructDesc = lowerStructType(ctx, subStructType);
        members.push_back(
            mlir::LLVM::TBAAMemberAttr::get(subStructDesc, member.getOffset()));
      } else {
        llvm::errs() << "decs: " << desc << "\n";
        llvm_unreachable("NYI");
      }
    }
    ret = mlir::LLVM::TBAATypeDescriptorAttr::get(
        ctx, structType.getId(),
        llvm::ArrayRef<mlir::LLVM::TBAAMemberAttr>(members));
    return structCache[structType] = ret;
  }

  mutable llvm::DenseMap<cir::TBAAStructTypeDescriptorAttr,
                         mlir::LLVM::TBAATypeDescriptorAttr>
      structCache;
};

std::unique_ptr<mlir::Pass> createTBAAPass() {
  return std::make_unique<TBAAPass>();
}

void populateCIRToLLVMPasses(mlir::OpPassManager &pm, bool useCCLowering) {
  populateCIRPreLoweringPasses(pm, useCCLowering);
  pm.addPass(createTBAAPass());
  pm.addPass(createConvertCIRToLLVMPass());
}

extern void registerCIRDialectTranslation(mlir::MLIRContext &context);

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule, LLVMContext &llvmCtx,
                             bool disableVerifier, bool disableCCLowering,
                             bool disableDebugInfo) {
  llvm::TimeTraceScope scope("lower from CIR to LLVM directly");

  mlir::MLIRContext *mlirCtx = theModule.getContext();
  mlir::PassManager pm(mlirCtx);
  populateCIRToLLVMPasses(pm, !disableCCLowering);

  // This is necessary to have line tables emitted and basic
  // debugger working. In the future we will add proper debug information
  // emission directly from our frontend.
  if (!disableDebugInfo) {
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }
  // FIXME(cir): this shouldn't be necessary. It's meant to be a temporary
  // workaround until we understand why some unrealized casts are being
  // emmited and how to properly avoid them.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.enableVerifier(!disableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  mlir::registerOpenMPDialectTranslation(*mlirCtx);
  registerCIRDialectTranslation(*mlirCtx);

  llvm::TimeTraceScope __scope("translateModuleToLLVMIR");

  auto ModuleName = theModule.getName();
  auto llvmModule = mlir::translateModuleToLLVMIR(
      theModule, llvmCtx, ModuleName ? *ModuleName : "CIRToLLVMModule");

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}
} // namespace direct
} // namespace cir
