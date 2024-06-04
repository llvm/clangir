//====- LoweringPrepareItaniumCXXABI.cpp - Itanium ABI specific code-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
//
// This file provides Itanium C++ ABI specific code
// that is used during LLVMIR lowering prepare.
//
//===--------------------------------------------------------------------===//

#include "LoweringPrepareItaniumCXXABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

using namespace cir;

cir::LoweringPrepareCXXABI *cir::LoweringPrepareCXXABI::createItaniumABI() {
  return new LoweringPrepareItaniumCXXABI();
}

static void buildBadCastCall(CIRBaseBuilderTy &builder, mlir::Location loc,
                             mlir::FlatSymbolRefAttr badCastFuncRef) {
  // TODO(cir): set the calling convention to __cxa_bad_cast.
  assert(!MissingFeatures::setCallingConv());

  builder.create<mlir::cir::CallOp>(loc, badCastFuncRef, mlir::ValueRange{});
  builder.create<mlir::cir::UnreachableOp>(loc);
  builder.clearInsertionPoint();
}

static mlir::Value buildDynamicCastAfterNullCheck(CIRBaseBuilderTy &builder,
                                                  mlir::cir::DynamicCastOp op) {
  auto loc = op->getLoc();
  auto srcValue = op.getSrc();
  auto castInfo = op.getInfo().value();

  // TODO(cir): consider address space
  assert(!MissingFeatures::addressSpace());

  auto srcPtr = builder.createBitcast(srcValue, builder.getVoidPtrTy());
  auto srcRtti = builder.getConstant(loc, castInfo.getSrcRtti());
  auto destRtti = builder.getConstant(loc, castInfo.getDestRtti());
  auto offsetHint = builder.getConstant(loc, castInfo.getOffsetHint());

  auto dynCastFuncRef = castInfo.getRuntimeFunc();
  mlir::Value dynCastFuncArgs[4] = {srcPtr, srcRtti, destRtti, offsetHint};

  // TODO(cir): set the calling convention for __dynamic_cast.
  assert(!MissingFeatures::setCallingConv());
  mlir::Value castedPtr =
      builder
          .create<mlir::cir::CallOp>(loc, dynCastFuncRef,
                                     builder.getVoidPtrTy(), dynCastFuncArgs)
          .getResult();

  assert(castedPtr.getType().isa<mlir::cir::PointerType>() &&
         "the return value of __dynamic_cast should be a ptr");

  /// C++ [expr.dynamic.cast]p9:
  ///   A failed cast to reference type throws std::bad_cast
  if (op.isRefcast()) {
    // Emit a cir.if that checks the casted value.
    mlir::Value castedValueIsNull = builder.createPtrIsNull(castedPtr);
    builder.create<mlir::cir::IfOp>(
        loc, castedValueIsNull, false, [&](mlir::OpBuilder &, mlir::Location) {
          buildBadCastCall(builder, loc, castInfo.getBadCastFunc());
        });
  }

  // Note that castedPtr is a void*. Cast it to a pointer to the destination
  // type before return.
  return builder.createBitcast(castedPtr, op.getType());
}

static mlir::Value
buildDynamicCastToVoidAfterNullCheck(CIRBaseBuilderTy &builder,
                                     clang::ASTContext &astCtx,
                                     mlir::cir::DynamicCastOp op) {
  auto loc = op.getLoc();
  bool vtableUsesRelativeLayout = op.getRelativeLayout();

  // TODO(cir): consider address space in this function.
  assert(!MissingFeatures::addressSpace());

  mlir::Type vtableElemTy;
  uint64_t vtableElemAlign;
  if (vtableUsesRelativeLayout) {
    vtableElemTy = builder.getSIntNTy(32);
    vtableElemAlign = 4;
  } else {
    const auto &targetInfo = astCtx.getTargetInfo();
    auto ptrdiffTy = targetInfo.getPtrDiffType(clang::LangAS::Default);
    auto ptrdiffTyIsSigned = clang::TargetInfo::isTypeSigned(ptrdiffTy);
    auto ptrdiffTyWidth = targetInfo.getTypeWidth(ptrdiffTy);

    vtableElemTy = mlir::cir::IntType::get(builder.getContext(), ptrdiffTyWidth,
                                           ptrdiffTyIsSigned);
    vtableElemAlign =
        llvm::divideCeil(targetInfo.getPointerAlign(clang::LangAS::Default), 8);
  }

  // Access vtable to get the offset from the given object to its containing
  // complete object.
  auto vtablePtrTy = builder.getPointerTo(vtableElemTy);
  auto vtablePtrPtr =
      builder.createBitcast(op.getSrc(), builder.getPointerTo(vtablePtrTy));
  auto vtablePtr = builder.createLoad(loc, vtablePtrPtr);
  auto offsetToTopSlotPtr = builder.create<mlir::cir::VTableAddrPointOp>(
      loc, vtablePtrTy, mlir::FlatSymbolRefAttr{}, vtablePtr,
      /*vtable_index=*/0, -2ULL);
  auto offsetToTop =
      builder.createAlignedLoad(loc, offsetToTopSlotPtr, vtableElemAlign);

  // Add the offset to the given pointer to get the cast result.
  // Cast the input pointer to a uint8_t* to allow pointer arithmetic.
  auto u8PtrTy = builder.getPointerTo(builder.getUIntNTy(8));
  auto srcBytePtr = builder.createBitcast(op.getSrc(), u8PtrTy);
  auto dstBytePtr = builder.create<mlir::cir::PtrStrideOp>(
      loc, u8PtrTy, srcBytePtr, offsetToTop);
  // Cast the result to a void*.
  return builder.createBitcast(dstBytePtr, builder.getVoidPtrTy());
}

mlir::Value
LoweringPrepareItaniumCXXABI::lowerDynamicCast(CIRBaseBuilderTy &builder,
                                               clang::ASTContext &astCtx,
                                               mlir::cir::DynamicCastOp op) {
  auto loc = op->getLoc();
  auto srcValue = op.getSrc();

  assert(!MissingFeatures::buildTypeCheck());

  if (op.isRefcast())
    return buildDynamicCastAfterNullCheck(builder, op);

  auto srcValueIsNotNull = builder.createPtrToBoolCast(srcValue);
  return builder
      .create<mlir::cir::TernaryOp>(
          loc, srcValueIsNotNull,
          [&](mlir::OpBuilder &, mlir::Location) {
            mlir::Value castedValue =
                op.isCastToVoid()
                    ? buildDynamicCastToVoidAfterNullCheck(builder, astCtx, op)
                    : buildDynamicCastAfterNullCheck(builder, op);
            builder.createYield(loc, castedValue);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(
                loc, builder.getNullPtr(op.getType(), loc).getResult());
          })
      .getResult();
}

mlir::Value LoweringPrepareItaniumCXXABI::lowerVAArg(
    CIRBaseBuilderTy &builder, mlir::cir::VAArgOp op,
    const ::cir::CIRDataLayout &datalayout) {
  // There is no generic cir lowering for var_arg, here we fail
  // so to prevent attempt of calling lowerVAArg for ItaniumCXXABI
  llvm_unreachable("NYI");
}
