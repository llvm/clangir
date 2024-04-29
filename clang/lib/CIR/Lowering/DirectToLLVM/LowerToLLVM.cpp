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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace cir;
using namespace llvm;

namespace cir {
namespace direct {

class CIRPtrStrideOpLowering
    : public mlir::OpConversionPattern<mlir::cir::PtrStrideOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::PtrStrideOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::PtrStrideOp ptrStrideOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *tc = getTypeConverter();
    const auto resultTy = tc->convertType(ptrStrideOp.getType());
    const auto elementTy = tc->convertType(ptrStrideOp.getElementTy());
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(ptrStrideOp, resultTy,
                                                   elementTy, adaptor.getBase(),
                                                   adaptor.getStride());

    return mlir::success();
  }
};

class CIRLoopOpLowering : public mlir::OpConversionPattern<mlir::cir::LoopOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::LoopOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOp loopOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (loopOp.getKind() != mlir::cir::LoopOpKind::For)
      llvm_unreachable("NYI");

    auto loc = loopOp.getLoc();

    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (loopOp->getResults().size() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    auto &condRegion = loopOp.getCond();
    auto &condFrontBlock = condRegion.front();

    auto &stepRegion = loopOp.getStep();
    auto &stepFrontBlock = stepRegion.front();
    auto &stepBackBlock = stepRegion.back();

    auto &bodyRegion = loopOp.getBody();
    auto &bodyFrontBlock = bodyRegion.front();
    auto &bodyBackBlock = bodyRegion.back();

    bool rewroteContinue = false;
    bool rewroteBreak = false;

    for (auto &bb : condRegion) {
      if (rewroteContinue && rewroteBreak)
        break;

      if (auto yieldOp = dyn_cast<mlir::cir::YieldOp>(bb.getTerminator())) {
        rewriter.setInsertionPointToEnd(yieldOp->getBlock());
        if (yieldOp.getKind().has_value()) {
          switch (yieldOp.getKind().value()) {
          case mlir::cir::YieldOpKind::Break:
          case mlir::cir::YieldOpKind::Fallthrough:
          case mlir::cir::YieldOpKind::NoSuspend:
            llvm_unreachable("None of these should be present");
          case mlir::cir::YieldOpKind::Continue:;
            rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
                yieldOp, yieldOp.getArgs(), &stepFrontBlock);
            rewroteContinue = true;
          }
        } else {
          rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
              yieldOp, yieldOp.getArgs(), continueBlock);
          rewroteBreak = true;
        }
      }
    }

    rewriter.inlineRegionBefore(condRegion, continueBlock);

    rewriter.inlineRegionBefore(stepRegion, continueBlock);

    if (auto stepYieldOp =
            dyn_cast<mlir::cir::YieldOp>(stepBackBlock.getTerminator())) {
      rewriter.setInsertionPointToEnd(stepYieldOp->getBlock());
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
          stepYieldOp, stepYieldOp.getArgs(), &bodyFrontBlock);
    } else {
      llvm_unreachable("What are we terminating with?");
    }

    rewriter.inlineRegionBefore(bodyRegion, continueBlock);

    if (auto bodyYieldOp =
            dyn_cast<mlir::cir::YieldOp>(bodyBackBlock.getTerminator())) {
      rewriter.setInsertionPointToEnd(bodyYieldOp->getBlock());
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
          bodyYieldOp, bodyYieldOp.getArgs(), &condFrontBlock);
    } else {
      llvm_unreachable("What are we terminating with?");
    }

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cir::BrOp>(loc, mlir::ValueRange(), &condFrontBlock);

    rewriter.replaceOp(loopOp, continueBlock->getArguments());

    return mlir::success();
  }
};

class CIRBrCondOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BrCondOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::LLVM::TruncOp>(
        brOp.getLoc(), rewriter.getI1Type(), condition);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
        brOp, i1Condition.getResult(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRCastOpLowering : public mlir::OpConversionPattern<mlir::cir::CastOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::CastOp>::OpConversionPattern;

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp castOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    switch (castOp.getKind()) {
    case mlir::cir::CastKind::array_to_ptrdecay: {
      const auto ptrTy = castOp.getType().cast<mlir::cir::PointerType>();
      auto sourceValue = adaptor.getOperands().front();
      auto targetType =
          getTypeConverter()->convertType(castOp->getResult(0).getType());
      auto elementTy = convertTy(ptrTy.getPointee());
      auto offset = llvm::SmallVector<mlir::LLVM::GEPArg>{0};
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
          castOp, targetType, elementTy, sourceValue, offset);
      break;
    }
    case mlir::cir::CastKind::int_to_bool: {
      auto zero = rewriter.create<mlir::cir::ConstantOp>(
          src.getLoc(), castOp.getSrc().getType(),
          mlir::cir::IntAttr::get(castOp.getSrc().getType(), 0));
      rewriter.replaceOpWithNewOp<mlir::cir::CmpOp>(
          castOp, mlir::cir::BoolType::get(getContext()),
          mlir::cir::CmpOpKind::ne, src, zero);
      break;
    }
    case mlir::cir::CastKind::integral: {
      auto dstType = castOp.getResult().getType().cast<mlir::cir::IntType>();
      auto srcType = castOp.getSrc().getType().dyn_cast<mlir::cir::IntType>();
      auto llvmSrcVal = adaptor.getOperands().front();
      auto llvmDstTy =
          getTypeConverter()->convertType(dstType).cast<mlir::IntegerType>();

      // Target integer is smaller: truncate source value.
      if (dstType.getWidth() < srcType.getWidth()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(castOp, llvmDstTy,
                                                         llvmSrcVal);
      } else {
        if (srcType.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(castOp, llvmDstTy,
                                                          llvmSrcVal);
      }
      break;
    }
    default:
      llvm_unreachable("NYI");
    }

    return mlir::success();
  }
};

class CIRIfLowering : public mlir::OpConversionPattern<mlir::cir::IfOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::IfOp ifOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    auto loc = ifOp.getLoc();

    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (ifOp->getResults().size() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region
    auto *thenBeforeBody = &ifOp.getThenRegion().front();
    auto *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<mlir::cir::YieldOp>(thenAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
          thenYieldOp, thenYieldOp.getArgs(), continueBlock);
    } else if (!dyn_cast<mlir::cir::ReturnOp>(thenAfterBody->getTerminator())) {
      llvm_unreachable("what are we terminating with?");
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Inline then region
    auto *elseBeforeBody = &ifOp.getElseRegion().front();
    auto *elseAfterBody = &ifOp.getElseRegion().back();
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), thenAfterBody);

    rewriter.setInsertionPointToEnd(currentBlock);
    auto trunc = rewriter.create<mlir::LLVM::TruncOp>(loc, rewriter.getI1Type(),
                                                      adaptor.getCondition());
    rewriter.create<mlir::LLVM::CondBrOp>(loc, trunc.getRes(), thenBeforeBody,
                                          elseBeforeBody);

    rewriter.setInsertionPointToEnd(elseAfterBody);
    if (auto elseYieldOp =
            dyn_cast<mlir::cir::YieldOp>(elseAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
          elseYieldOp, elseYieldOp.getArgs(), continueBlock);
    } else if (!dyn_cast<mlir::cir::ReturnOp>(elseAfterBody->getTerminator())) {
      llvm_unreachable("what are we terminating with?");
    }

    rewriter.replaceOp(ifOp, continueBlock->getArguments());

    return mlir::success();
  }
};

class CIRScopeOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ScopeOp> {
public:
  using OpConversionPattern<mlir::cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ScopeOp scopeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = scopeOp.getLoc();

    // Split the current block before the ScopeOp to create the inlining point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (scopeOp.getNumResults() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline body region.
    auto *beforeBody = &scopeOp.getRegion().front();
    auto *afterBody = &scopeOp.getRegion().back();
    rewriter.inlineRegionBefore(scopeOp.getRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    // TODO(CIR): stackSaveOp
    // auto stackSaveOp = rewriter.create<mlir::LLVM::StackSaveOp>(
    //     loc, mlir::LLVM::LLVMPointerType::get(
    //              mlir::IntegerType::get(scopeOp.getContext(), 8)));
    rewriter.create<mlir::cir::BrOp>(loc, mlir::ValueRange(), beforeBody);

    // Replace the scopeop return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    auto yieldOp = cast<mlir::cir::YieldOp>(afterBody->getTerminator());
    auto branchOp = rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        yieldOp, yieldOp.getArgs(), continueBlock);

    // // Insert stack restore before jumping out of the body of the region.
    rewriter.setInsertionPoint(branchOp);
    // TODO(CIR): stackrestore?
    // rewriter.create<mlir::LLVM::StackRestoreOp>(loc, stackSaveOp);

    // Replace the op with values return from the body region.
    rewriter.replaceOp(scopeOp, continueBlock->getArguments());

    return mlir::success();
  }
};

class CIRReturnLowering
    : public mlir::OpConversionPattern<mlir::cir::ReturnOp> {
public:
  using OpConversionPattern<mlir::cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

struct ConvertCIRToLLVMPass
    : public mlir::PassWrapper<ConvertCIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::DLTIDialect,
                    mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-llvm"; }
};

class CIRCallLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
public:
  using OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 8> llvmResults;
    auto cirResults = op.getResultTypes();

    if (getTypeConverter()->convertTypes(cirResults, llvmResults).failed())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, llvmResults, op.getCalleeAttr(), adaptor.getOperands());
    return mlir::success();
  }
};

class CIRAllocaLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
public:
  using OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto elementTy = getTypeConverter()->convertType(op.getAllocaType());

    mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), typeConverter->convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

    auto resultTy = mlir::LLVM::LLVMPointerType::get(getContext());

    rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(
        op, resultTy, elementTy, one, op.getAlignmentAttr().getInt());
    return mlir::LogicalResult::success();
  }
};

class CIRLoadLowering : public mlir::OpConversionPattern<mlir::cir::LoadOp> {
public:
  using OpConversionPattern<mlir::cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const auto llvmTy =
        getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, llvmTy,
                                                    adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
public:
  using OpConversionPattern<mlir::cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(),
                                                     adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRConstantLowering
    : public mlir::OpConversionPattern<mlir::cir::ConstantOp> {
public:
  using OpConversionPattern<mlir::cir::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Attribute attr = op.getValue();

    if (op.getType().isa<mlir::cir::BoolType>()) {
      if (op.getValue() ==
          mlir::cir::BoolAttr::get(
              getContext(), ::mlir::cir::BoolType::get(getContext()), true))
        attr = mlir::BoolAttr::get(getContext(), true);
      else
        attr = mlir::BoolAttr::get(getContext(), false);
    } else if (op.getType().isa<mlir::cir::IntType>()) {
      attr = rewriter.getIntegerAttr(
          typeConverter->convertType(op.getType()),
          op.getValue().cast<mlir::cir::IntAttr>().getValue());
    } else if (op.getType().isa<mlir::FloatType>()) {
      attr = op.getValue();
    } else
      return op.emitError("unsupported constant type");

    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, getTypeConverter()->convertType(op.getType()), attr);

    return mlir::success();
  }
};

class CIRFuncLowering : public mlir::OpConversionPattern<mlir::cir::FuncOp> {
public:
  using OpConversionPattern<mlir::cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto fnType = op.getFunctionType();
    mlir::TypeConverter::SignatureConversion signatureConversion(
        fnType.getNumInputs());

    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = typeConverter->convertType(argType.value());
      if (!convertedType)
        return mlir::failure();
      signatureConversion.addInputs(argType.index(), convertedType);
    }

    mlir::Type resultType;
    if (fnType.getNumResults() == 1) {
      resultType = getTypeConverter()->convertType(fnType.getResult(0));
      if (!resultType)
        return mlir::failure();
    }

    // Create the LLVM function operation.
    auto llvmFnTy = mlir::LLVM::LLVMFunctionType::get(
        resultType ? resultType : mlir::LLVM::LLVMVoidType::get(getContext()),
        signatureConversion.getConvertedTypes(),
        /*isVarArg=*/fnType.isVarArg());
    auto fn = rewriter.create<mlir::LLVM::LLVMFuncOp>(op.getLoc(), op.getName(),
                                                      llvmFnTy);

    rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
    if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                           &signatureConversion)))
      return mlir::failure();

    rewriter.eraseOp(op);

    return mlir::LogicalResult::success();
  }
};

mlir::DenseElementsAttr
convertToDenseElementsAttr(mlir::cir::ConstArrayAttr attr, mlir::Type type) {
  auto values = llvm::SmallVector<mlir::APInt, 8>{};
  auto arrayAttr = attr.getElts().dyn_cast<mlir::ArrayAttr>();
  assert(arrayAttr && "expected array here");
  for (auto element : arrayAttr)
    values.push_back(element.cast<mlir::cir::IntAttr>().getValue());
  return mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({(int64_t)values.size()}, type),
      llvm::ArrayRef(values));
}

std::optional<mlir::Attribute>
lowerConstArrayAttr(mlir::cir::ConstArrayAttr constArr,
                    const mlir::TypeConverter *converter) {

  // Ensure ConstArrayAttr has a type.
  auto typedConstArr = constArr.dyn_cast<mlir::TypedAttr>();
  assert(typedConstArr && "cir::ConstArrayAttr is not a mlir::TypedAttr");

  // Ensure ConstArrayAttr type is a ArrayType.
  auto cirArrayType = typedConstArr.getType().dyn_cast<mlir::cir::ArrayType>();
  assert(cirArrayType && "cir::ConstArrayAttr is not a cir::ArrayType");

  // Is a ConstArrayAttr with an cir::ArrayType: fetch element type.
  auto type = cirArrayType.getEltType();

  if (type.isa<mlir::cir::IntType>())
    return convertToDenseElementsAttr(constArr, converter->convertType(type));

  return std::nullopt;
}

mlir::LLVM::Linkage convertLinkage(mlir::cir::GlobalLinkageKind linkage) {
  using CIR = mlir::cir::GlobalLinkageKind;
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

class CIRGetGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GetGlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(op.getType());
    auto symbol = op.getName();
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, type, symbol);
    return mlir::success();
  }
};

class CIRGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    // Fetch required values to create LLVM op.
    auto llvmType = getTypeConverter()->convertType(op.getSymType());
    auto isConst = op.getConstant();
    auto linkage = convertLinkage(op.getLinkage());
    auto symbol = op.getSymName();
    auto init = op.getInitialValue();

    // Check for missing funcionalities.
    if (!init.has_value()) {
      op.emitError() << "uninitialized globals are not yet supported.";
      return mlir::failure();
    }

    // Initializer is a constant array: convert it to a compatible llvm init.
    if (auto constArr = init.value().dyn_cast<mlir::cir::ConstArrayAttr>()) {
      if (auto attr = constArr.getElts().dyn_cast<mlir::StringAttr>()) {
        init = rewriter.getStringAttr(attr.getValue());
      } else if (auto attr = constArr.getElts().dyn_cast<mlir::ArrayAttr>()) {
        if (!(init = lowerConstArrayAttr(constArr, getTypeConverter()))) {
          op.emitError()
              << "unsupported lowering for #cir.const_array with element type "
              << op.getSymType();
          return mlir::failure();
        }
      } else {
        op.emitError()
            << "unsupported lowering for #cir.const_array with value "
            << constArr.getElts();
        return mlir::failure();
      }
    } else if (llvm::isa<mlir::FloatAttr>(init.value())) {
      // Nothing to do since LLVM already supports these types as initializers.
    }
    // Initializer is a constant integer: convert to MLIR builtin constant.
    else if (auto intAttr = init.value().dyn_cast<mlir::cir::IntAttr>()) {
      init = rewriter.getIntegerAttr(llvmType, intAttr.getValue());
    }
    // Initializer is a global: load global value in initializer block.
    else if (auto attr = init.value().dyn_cast<mlir::FlatSymbolRefAttr>()) {
      auto newGlobalOp = rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
          op, llvmType, isConst, linkage, symbol, mlir::Attribute());
      mlir::OpBuilder::InsertionGuard guard(rewriter);

      // Create initializer block.
      auto *newBlock = new mlir::Block();
      newGlobalOp.getRegion().push_back(newBlock);

      // Fetch global used as initializer.
      auto sourceSymbol =
          dyn_cast<mlir::LLVM::GlobalOp>(mlir::SymbolTable::lookupSymbolIn(
              op->getParentOfType<mlir::ModuleOp>(), attr.getValue()));

      // Load and return the initializer value.
      rewriter.setInsertionPointToEnd(newBlock);
      auto addressOfOp = rewriter.create<mlir::LLVM::AddressOfOp>(
          op->getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()),
          sourceSymbol.getSymName());
      llvm::SmallVector<mlir::LLVM::GEPArg> offset{0};
      auto gepOp = rewriter.create<mlir::LLVM::GEPOp>(
          op->getLoc(), llvmType, sourceSymbol.getType(),
          addressOfOp.getResult(), offset);
      rewriter.create<mlir::LLVM::ReturnOp>(op->getLoc(), gepOp.getResult());

      return mlir::success();
    } else {
      op.emitError() << "usupported initializer '" << init.value() << "'";
      return mlir::failure();
    }

    // Rewrite op.
    rewriter.replaceOpWithNewOp<mlir::LLVM::GlobalOp>(
        op, llvmType, isConst, linkage, symbol, init.value());
    return mlir::success();
  }
};

class CIRUnaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::UnaryOp> {
public:
  using OpConversionPattern<mlir::cir::UnaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type type = op.getInput().getType();
    assert(type.isa<mlir::cir::IntType>() && "operand type not supported yet");

    auto llvmInType = adaptor.getInput().getType();
    auto llvmType = getTypeConverter()->convertType(op.getType());

    switch (op.getKind()) {
    case mlir::cir::UnaryOpKind::Inc: {
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), llvmInType, mlir::IntegerAttr::get(llvmInType, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmType,
                                                     adaptor.getInput(), One);
      break;
    }
    case mlir::cir::UnaryOpKind::Dec: {
      auto One = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), llvmInType, mlir::IntegerAttr::get(llvmInType, 1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType,
                                                     adaptor.getInput(), One);
      break;
    }
    case mlir::cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, adaptor.getInput());
      break;
    }
    case mlir::cir::UnaryOpKind::Minus: {
      auto Zero = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), llvmInType, mlir::IntegerAttr::get(llvmInType, 0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmType, Zero,
                                                     adaptor.getInput());
      break;
    }
    case mlir::cir::UnaryOpKind::Not: {
      auto MinusOne = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), llvmType, mlir::IntegerAttr::get(llvmType, -1));
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, llvmType, MinusOne,
                                                     adaptor.getInput());
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBinOpLowering : public mlir::OpConversionPattern<mlir::cir::BinOp> {
public:
  using OpConversionPattern<mlir::cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert((op.getLhs().getType() == op.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type type = op.getRhs().getType();
    assert((type.isa<mlir::cir::IntType, mlir::FloatType>()) &&
           "operand type not supported yet");

    auto llvmTy = getTypeConverter()->convertType(op.getType());
    auto rhs = adaptor.getRhs();
    auto lhs = adaptor.getLhs();

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (type.isa<mlir::cir::IntType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::AddOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FAddOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Sub:
      if (type.isa<mlir::cir::IntType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::SubOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FSubOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Mul:
      if (type.isa<mlir::cir::IntType>())
        rewriter.replaceOpWithNewOp<mlir::LLVM::MulOp>(op, llvmTy, lhs, rhs);
      else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FMulOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Div:
      if (auto ty = type.dyn_cast<mlir::cir::IntType>()) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::UDivOp>(op, llvmTy, lhs, rhs);
        else
          llvm_unreachable("signed integer division binop lowering NYI");
      } else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FDivOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Rem:
      if (auto ty = type.dyn_cast<mlir::cir::IntType>()) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::URemOp>(op, llvmTy, lhs, rhs);
        else
          llvm_unreachable("signed integer remainder binop lowering NYI");
      } else
        rewriter.replaceOpWithNewOp<mlir::LLVM::FRemOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::LLVM::XOrOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Shl:
      rewriter.replaceOpWithNewOp<mlir::LLVM::ShlOp>(op, llvmTy, lhs, rhs);
      break;
    case mlir::cir::BinOpKind::Shr:
      if (auto ty = type.dyn_cast<mlir::cir::IntType>()) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::LLVM::LShrOp>(op, llvmTy, lhs, rhs);
        else
          llvm_unreachable("signed integer shift binop lowering NYI");
        break;
      }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<mlir::cir::CmpOp> {
public:
  using OpConversionPattern<mlir::cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp cmpOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getLhs().getType();
    auto i1Type =
        mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless);
    auto destType = getTypeConverter()->convertType(cmpOp.getType());

    switch (adaptor.getKind()) {
    case mlir::cir::CmpOpKind::gt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::ugt;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ugt),
            adaptor.getLhs(), adaptor.getRhs(),
            // TODO(CIR): These fastmath flags need to not be defaulted.
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ge: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::uge;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::uge),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::lt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::ult;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ult),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::le: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::LLVM::ICmpPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::LLVM::ICmpPredicate::ule;
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ule),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::eq: {
      if (type.isa<mlir::IntegerType>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::ICmpPredicate::eq),
            adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::ueq),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ne: {
      if (type.isa<mlir::IntegerType>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::ICmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::ICmpPredicate::ne),
            adaptor.getLhs(), adaptor.getRhs());

        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else if (type.isa<mlir::FloatType>()) {
        auto cmp = rewriter.create<mlir::LLVM::FCmpOp>(
            cmpOp.getLoc(), i1Type,
            mlir::LLVM::FCmpPredicateAttr::get(getContext(),
                                               mlir::LLVM::FCmpPredicate::une),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::LLVM::FastmathFlagsAttr::get(cmpOp.getContext(), {}));
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(cmpOp, destType,
                                                        cmp.getRes());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpRewritePattern<mlir::cir::BrOp> {
public:
  using OpRewritePattern<mlir::cir::BrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(op, op.getDestOperands(),
                                                  op.getDest());
    return mlir::LogicalResult::success();
  }
};

void populateCIRToLLVMConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRBrOpLowering, CIRReturnLowering>(patterns.getContext());
  patterns.add<CIRCmpOpLowering, CIRLoopOpLowering, CIRBrCondOpLowering,
               CIRPtrStrideOpLowering, CIRCallLowering, CIRUnaryOpLowering,
               CIRBinOpLowering, CIRLoadLowering, CIRConstantLowering,
               CIRStoreLowering, CIRAllocaLowering, CIRFuncLowering,
               CIRScopeOpLowering, CIRCastOpLowering, CIRIfLowering,
               CIRGlobalOpLowering, CIRGetGlobalOpLowering>(
      converter, patterns.getContext());
}

namespace {
void prepareTypeConverter(mlir::LLVMTypeConverter &converter) {
  converter.addConversion([&](mlir::cir::PointerType type) -> mlir::Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&](mlir::cir::ArrayType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::LLVM::LLVMArrayType::get(ty, type.getSize());
  });
  converter.addConversion([&](mlir::cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8,
                                  mlir::IntegerType::Signless);
  });
  converter.addConversion([&](mlir::cir::IntType type) -> mlir::Type {
    // LLVM doesn't work with signed types, so we drop the CIR signs here.
    return mlir::IntegerType::get(type.getContext(), type.getWidth());
  });
}
} // namespace

void ConvertCIRToLLVMPass::runOnOperation() {
  auto module = getOperation();

  mlir::LLVMTypeConverter converter(&getContext());
  prepareTypeConverter(converter);

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRToLLVMConversionPatterns(patterns, converter);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);

  mlir::ConversionTarget target(getContext());
  using namespace mlir::cir;
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
                    // ,LoopOp
                    // ,ReturnOp
                    // ,StoreOp
                    // ,YieldOp
                    >();
  // clang-format on
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::BuiltinDialect, mlir::cir::CIRDialect,
                           mlir::func::FuncDialect>();

  getOperation()->removeAttr("cir.sob");

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule,
                             std::unique_ptr<mlir::MLIRContext> mlirCtx,
                             LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createConvertCIRToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);

  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}
} // namespace direct
} // namespace cir
