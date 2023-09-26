//====- LowerCIRToMLIR.cpp - Lowering from CIR to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/Sequence.h"

using namespace cir;
using namespace llvm;

namespace cir {

// Forward declaration from LowerMLIRToLLVM.cpp
std::unique_ptr<mlir::Pass> createConvertMLIRToLLVMPass();

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

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRCallLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
public:
  using OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, mlir::SymbolRefAttr::get(op), types, adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRAllocaLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
public:
  using OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the type being allocated (the pointee type)
    auto allocaType = op.getAllocaType();
    auto mlirType = getTypeConverter()->convertType(allocaType);
    if (!mlirType) {
      return mlir::LogicalResult::failure();
    }
    
    // Create a 0-D memref (scalar) for the allocated type
    auto memreftype = mlir::MemRefType::get({}, mlirType);
    
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memreftype,
                                                        op.getAlignmentAttr());
    return mlir::LogicalResult::success();
  }
};

class CIRLoadLowering : public mlir::OpConversionPattern<mlir::cir::LoadOp> {
public:
  using OpConversionPattern<mlir::cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
public:
  using OpConversionPattern<mlir::cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, adaptor.getValue(),
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
    auto ty = getTypeConverter()->convertType(op.getType());
    mlir::TypedAttr value;
    
    if (mlir::isa<mlir::cir::BoolType>(op.getType())) {
      auto boolValue = mlir::cast<mlir::cir::BoolAttr>(op.getValue());
      value = rewriter.getIntegerAttr(ty, boolValue.getValue());
    } else if (auto cirIntAttr = mlir::dyn_cast<mlir::cir::IntAttr>(op.getValue())) {
      value = rewriter.getIntegerAttr(ty, cirIntAttr.getValue());
    } else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(op.getValue())) {
      // Handle floating-point constants
      value = rewriter.getFloatAttr(ty, floatAttr.getValueAsDouble());
    } else {
      return mlir::LogicalResult::failure();
    }
    
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, ty, value);
    return mlir::LogicalResult::success();
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
        return failure();
      signatureConversion.addInputs(argType.index(), convertedType);
    }

    SmallVector<mlir::Type> resultTypes;
    // Only convert return type if the function is not void
    if (!fnType.isVoid()) {
      auto resultType = getTypeConverter()->convertType(fnType.getReturnType());
      if (!resultType)
        return failure();
      resultTypes.push_back(resultType);
    }

    auto fn = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getName(),
        rewriter.getFunctionType(signatureConversion.getConvertedTypes(),
                                 resultTypes));

    rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());
    if (failed(rewriter.convertRegionTypes(&fn.getBody(), *typeConverter,
                                           &signatureConversion)))
      return mlir::failure();

    rewriter.eraseOp(op);

    return mlir::LogicalResult::success();
  }
};

class CIRUnaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::UnaryOp> {
public:
  using OpConversionPattern<mlir::cir::UnaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto type = getTypeConverter()->convertType(op.getType());

    switch (op.getKind()) {
    case mlir::cir::UnaryOpKind::Inc: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, type, input, One);
      break;
    }
    case mlir::cir::UnaryOpKind::Dec: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, input, One);
      break;
    }
    case mlir::cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, op.getInput());
      break;
    }
    case mlir::cir::UnaryOpKind::Minus: {
      auto Zero = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 0));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, Zero, input);
      break;
    }
    case mlir::cir::UnaryOpKind::Not: {
      auto MinusOne = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, -1));
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, type, MinusOne,
                                                       input);
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
    assert((adaptor.getLhs().getType() == adaptor.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
    assert((mlir::isa<mlir::IntegerType>(mlirType) ||
            mlir::isa<mlir::FloatType>(mlirType)) &&
           "operand type not supported yet");

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (mlir::isa<mlir::IntegerType>(mlirType))
        rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Sub:
      if (mlir::isa<mlir::IntegerType>(mlirType))
        rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Mul:
      if (mlir::isa<mlir::IntegerType>(mlirType))
        rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Div:
      if (mlir::isa<mlir::IntegerType>(mlirType)) {
        if (mlirType.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          llvm_unreachable("integer mlirType not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Rem:
      if (mlir::isa<mlir::IntegerType>(mlirType)) {
        if (mlirType.isSignlessInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          llvm_unreachable("integer mlirType not supported in CIR yet");
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<mlir::cir::CmpOp> {
public:
  using OpConversionPattern<mlir::cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getLhs().getType();
    auto integerType =
        mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless);

    mlir::Value mlirResult;
    switch (op.getKind()) {
    case mlir::cir::CmpOpKind::gt: {
      if (mlir::isa<mlir::IntegerType>(type)) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ugt;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (mlir::isa<mlir::FloatType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGT),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ge: {
      if (mlir::isa<mlir::IntegerType>(type)) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::uge;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (mlir::isa<mlir::FloatType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGE),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::lt: {
      if (mlir::isa<mlir::IntegerType>(type)) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ult;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (mlir::isa<mlir::FloatType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULT),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::le: {
      if (mlir::isa<mlir::IntegerType>(type)) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ule;
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (mlir::isa<mlir::FloatType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULE),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::eq: {
      if (mlir::isa<mlir::IntegerType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::eq),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (mlir::isa<mlir::FloatType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UEQ),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ne: {
      if (mlir::isa<mlir::IntegerType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::ne),
            adaptor.getLhs(), adaptor.getRhs());
      } else if (mlir::isa<mlir::FloatType>(type)) {
        mlirResult = rewriter.create<mlir::arith::CmpFOp>(
            op.getLoc(), integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UNE),
            adaptor.getLhs(), adaptor.getRhs(),
            mlir::arith::FastMathFlagsAttr::get(
                getContext(), mlir::arith::FastMathFlags::none));
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    }

    auto converted_type = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOp(op, mlirResult);
    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpConversionPattern<mlir::cir::BrOp> {
public:
  using OpConversionPattern<mlir::cir::BrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest());
    return mlir::LogicalResult::success();
  }
};

class CIRBrCondOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BrCondOp> {
public:
  using OpConversionPattern<mlir::cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrCondOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCond(), op.getDestTrue(), op.getDestFalse());
    return mlir::LogicalResult::success();
  }
};

class CIRTernaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::TernaryOp> {
public:
  using OpConversionPattern<mlir::cir::TernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For now, lower ternary to if-then-else pattern
    // TODO: This is a simplified lowering that doesn't properly handle the
    // regions A proper implementation would need to inline the regions and
    // extract the yielded values
    return mlir::LogicalResult::failure();
  }
};

class CIRYieldOpLowering
    : public mlir::OpConversionPattern<mlir::cir::YieldOp> {
public:
  using OpConversionPattern<mlir::cir::YieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // For yields in memref.alloca_scope, convert to memref.alloca_scope.return
    if (op->getParentOp() &&
        isa<mlir::memref::AllocaScopeOp>(op->getParentOp())) {
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(
          op, adaptor.getOperands());
      return mlir::LogicalResult::success();
    }

    // Otherwise, convert to scf.yield
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRIfOpLowering : public mlir::OpConversionPattern<mlir::cir::IfOp> {
public:
  using OpConversionPattern<mlir::cir::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::IfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::scf::IfOp ifOp;
    // CIR IfOp doesn't have results, it's a void operation
    ifOp =
        rewriter.create<mlir::scf::IfOp>(op.getLoc(), adaptor.getCondition());

    auto inlineIfCase = [&](mlir::Region &ifCase, mlir::Region &cirCase) {
      if (cirCase.empty())
        return;

      rewriter.inlineRegionBefore(cirCase, ifCase, ifCase.end());
    };

    inlineIfCase(ifOp.getThenRegion(), op.getThenRegion());
    inlineIfCase(ifOp.getElseRegion(), op.getElseRegion());

    rewriter.replaceOp(op, ifOp.getResults());
    return mlir::LogicalResult::success();
  }
};

class CIRLoopOpLowering : public mlir::OpConversionPattern<mlir::cir::LoopOp> {
public:
  using OpConversionPattern<mlir::cir::LoopOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto whileOp = rewriter.create<mlir::scf::WhileOp>(
        op.getLoc(), mlir::TypeRange{}, mlir::ValueRange{});

    auto before = rewriter.createBlock(&whileOp.getBefore());
    rewriter.setInsertionPointToStart(before);
    auto conditionOp = rewriter.create<mlir::cir::ConstantOp>(
        op.getLoc(), mlir::cir::BoolType::get(getContext()),
        mlir::cir::BoolAttr::get(getContext(),
                                 mlir::cir::BoolType::get(getContext()), true));
    rewriter.create<mlir::scf::ConditionOp>(op.getLoc(), conditionOp.getRes(),
                                            mlir::ValueRange{});

    auto after = rewriter.createBlock(&whileOp.getAfter());
    rewriter.inlineRegionBefore(op.getBody(), whileOp.getAfter(),
                                whileOp.getAfter().end());
    auto mergedAfter = &whileOp.getAfter().back();
    rewriter.mergeBlocks(after, mergedAfter, {});

    rewriter.replaceOp(op, whileOp.getResults());
    return mlir::LogicalResult::success();
  }
};

class CIRScopeOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ScopeOp> {
public:
  using OpConversionPattern<mlir::cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ScopeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Check if the scope is empty (no operations)
    if (op.getScopeRegion().empty() ||
        (op.getScopeRegion().front().empty() ||
         (op.getScopeRegion().front().getOperations().size() == 1 &&
          isa<mlir::cir::YieldOp>(op.getScopeRegion().front().front())))) {
      // Drop empty scopes
      rewriter.eraseOp(op);
      return mlir::LogicalResult::success();
    }

    // For scopes without results, use memref.alloca_scope
    if (op.getResults().empty()) {
      auto allocaScope = rewriter.create<mlir::memref::AllocaScopeOp>(
          op.getLoc(), mlir::TypeRange{});
      rewriter.inlineRegionBefore(op.getScopeRegion(),
                                  allocaScope.getBodyRegion(),
                                  allocaScope.getBodyRegion().end());
      rewriter.eraseOp(op);
    } else {
      // For scopes with results, use scf.execute_region
      SmallVector<mlir::Type> types;
      if (mlir::failed(
              getTypeConverter()->convertTypes(op->getResultTypes(), types)))
        return mlir::failure();
      auto exec =
          rewriter.create<mlir::scf::ExecuteRegionOp>(op.getLoc(), types);
      rewriter.inlineRegionBefore(op.getScopeRegion(), exec.getRegion(),
                                  exec.getRegion().end());
      rewriter.replaceOp(op, exec.getResults());
    }
    return mlir::LogicalResult::success();
  }
};

class CIRCastOpLowering : public mlir::OpConversionPattern<mlir::cir::CastOp> {
public:
  using OpConversionPattern<mlir::cir::CastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getSrc().getType();
    auto dstType = getTypeConverter()->convertType(op.getResult().getType());

    switch (op.getKind()) {
    case mlir::cir::CastKind::int_to_bool: {
      auto zero = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), srcType, rewriter.getIntegerAttr(srcType, 0));
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
          op, dstType,
          mlir::arith::CmpIPredicateAttr::get(getContext(),
                                              mlir::arith::CmpIPredicate::ne),
          adaptor.getSrc(), zero);
      return mlir::LogicalResult::success();
    }
    case mlir::cir::CastKind::integral: {
      rewriter.replaceOpWithNewOp<mlir::arith::TruncIOp>(op, dstType,
                                                         adaptor.getSrc());
      return mlir::LogicalResult::success();
    }
    case mlir::cir::CastKind::floating: {
      rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, dstType,
                                                         adaptor.getSrc());
      return mlir::LogicalResult::success();
    }
    case mlir::cir::CastKind::int_to_float: {
      rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(op, dstType,
                                                         adaptor.getSrc());
      return mlir::LogicalResult::success();
    }
    case mlir::cir::CastKind::float_to_int: {
      rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(op, dstType,
                                                         adaptor.getSrc());
      return mlir::LogicalResult::success();
    }
    case mlir::cir::CastKind::bool_to_int: {
      rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, dstType,
                                                        adaptor.getSrc());
      return mlir::LogicalResult::success();
    }
    case mlir::cir::CastKind::array_to_ptrdecay: {
      // TODO: this is not correct, we shoudl first convert the array to a
      // memref and then extract the aligned pointer.
      rewriter.replaceOp(op, adaptor.getSrc());
      return mlir::LogicalResult::success();
    }
    case mlir::cir::CastKind::bitcast: {
      rewriter.replaceOpWithNewOp<mlir::arith::BitcastOp>(op, dstType,
                                                          adaptor.getSrc());
      return mlir::LogicalResult::success();
    }
    }

    return mlir::LogicalResult::failure();
  }
};

class CIRGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(op.getSymType());
    auto memrefType = mlir::MemRefType::get({}, type);
    auto newGlobal = rewriter.create<mlir::memref::GlobalOp>(
        op.getLoc(), op.getSymName(), op.getSymVisibilityAttr(), memrefType,
        op.getInitialValueAttr(), static_cast<bool>(op.getConstantAttr()),
        nullptr);

    rewriter.eraseOp(op);
    return mlir::LogicalResult::success();
  }
};

class CIRGetGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GetGlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = getTypeConverter()->convertType(op.getAddr().getType());
    rewriter.replaceOpWithNewOp<mlir::memref::GetGlobalOp>(op, type,
                                                           op.getName());
    return mlir::LogicalResult::success();
  }
};

// Create a custom type converter that properly handles CIR types
class CIRTypeConverter : public mlir::TypeConverter {
public:
  CIRTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](mlir::cir::IntType type) -> std::optional<mlir::Type> {
      return mlir::IntegerType::get(type.getContext(), type.getWidth());
    });
    addConversion([&](mlir::cir::BoolType type) -> std::optional<mlir::Type> {
      return mlir::IntegerType::get(type.getContext(), 8,
                                    mlir::IntegerType::Signless);
    });
    addConversion([&](mlir::cir::PointerType type) -> std::optional<mlir::Type> {
      // For pointers, create a memref with the pointee type
      auto pointeeType = convertType(type.getPointee());
      if (!pointeeType)
        return std::nullopt;
      // For pointers to scalars, create a scalar memref
      // For pointers to arrays, create a dynamic memref
      if (isa<mlir::cir::ArrayType>(type.getPointee()))
        return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, pointeeType);
      else
        return mlir::MemRefType::get({}, pointeeType);
    });
    addConversion([&](mlir::cir::ArrayType type) -> std::optional<mlir::Type> {
      auto elementType = convertType(type.getEltType());
      if (!elementType)
        return std::nullopt;
      return mlir::MemRefType::get({static_cast<int64_t>(type.getSize())}, elementType);
    });
    // Add float type conversions
    addConversion([](mlir::FloatType type) -> std::optional<mlir::Type> {
      return type; // Float types are already MLIR native
    });
  }
};

void ConvertCIRToMLIRPass::runOnOperation() {
  mlir::MLIRContext *context = &getContext();
  mlir::ModuleOp theModule = getOperation();

  auto converter = CIRTypeConverter{};
  mlir::ConversionTarget target(*context);

  target.addLegalDialect<mlir::BuiltinDialect, mlir::func::FuncDialect,
                         mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
                         mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect>();

  target.addIllegalDialect<mlir::cir::CIRDialect>();

  mlir::RewritePatternSet patterns(context);
  patterns.add<CIRReturnLowering, CIRFuncLowering, CIRCallLowering,
               CIRAllocaLowering, CIRLoadLowering, CIRStoreLowering,
               CIRConstantLowering, CIRUnaryOpLowering, CIRBinOpLowering,
               CIRCmpOpLowering, CIRBrOpLowering, CIRBrCondOpLowering,
               CIRTernaryOpLowering, CIRYieldOpLowering, CIRIfOpLowering,
               CIRLoopOpLowering, CIRScopeOpLowering, CIRCastOpLowering,
               CIRGlobalOpLowering, CIRGetGlobalOpLowering>(converter, context);

  if (mlir::failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass() {
  return std::make_unique<ConvertCIRToMLIRPass>();
}

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp theModule,
                                  mlir::MLIRContext *mlirCtx) {
  mlir::PassManager pm(mlirCtx);
  pm.addPass(createConvertCIRToMLIRPass());

  if (mlir::failed(pm.run(theModule))) {
    return nullptr;
  }

  return theModule;
}

std::unique_ptr<llvm::Module>
lowerFromCIRToMLIRToLLVMIR(mlir::ModuleOp theModule,
                           std::unique_ptr<mlir::MLIRContext> mlirCtx,
                           llvm::LLVMContext &llvmCtx) {
  // First lower from CIR to MLIR
  mlir::PassManager pm(mlirCtx.get());
  pm.addPass(createConvertCIRToMLIRPass());

  if (mlir::failed(pm.run(theModule))) {
    return nullptr;
  }

  // Then lower from MLIR to LLVM
  pm.clear();
  pm.addPass(createConvertMLIRToLLVMPass());

  if (mlir::failed(pm.run(theModule))) {
    return nullptr;
  }

  // Finally export to LLVM IR
  return mlir::translateModuleToLLVMIR(theModule, llvmCtx);
}

} // namespace cir
