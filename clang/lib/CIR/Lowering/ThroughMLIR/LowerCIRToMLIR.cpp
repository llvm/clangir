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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

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
                    mlir::scf::SCFDialect, mlir::math::MathDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRCallOpLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
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

class CIRAllocaOpLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
public:
  using OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getAllocaType();
    auto mlirType = getTypeConverter()->convertType(type);

    // FIXME: Some types can not be converted yet (e.g. struct)
    if (!mlirType)
      return mlir::LogicalResult::failure();
    
    auto memreftype = mlir::MemRefType::get({}, mlirType);
    
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memreftype,
                                                        op.getAlignmentAttr());
    return mlir::LogicalResult::success();
  }
};

class CIRLoadOpLowering : public mlir::OpConversionPattern<mlir::cir::LoadOp> {
public:
  using OpConversionPattern<mlir::cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRStoreOpLowering
    : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
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

class CIRCosOpLowering : public mlir::OpConversionPattern<mlir::cir::CosOp> {
public:
  using OpConversionPattern<mlir::cir::CosOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CosOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto convertedType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::math::CosOp>(op, convertedType, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRConstantOpLowering
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
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(op.getType())) {
      auto fpAttr = mlir::cast<mlir::cir::FPAttr>(op.getValue());
      auto apFloat = fpAttr.getValue();
      
      // Handle specific floating-point types explicitly
      if (mlir::isa<mlir::cir::SingleType>(op.getType())) {
        auto f32Type = mlir::Float32Type::get(rewriter.getContext());
        // Convert APFloat to f32 precision
        bool losesInfo = false;
        llvm::APFloat f32Float = apFloat;
        f32Float.convert(llvm::APFloat::IEEEsingle(), llvm::RoundingMode::NearestTiesToEven, &losesInfo);
        value = rewriter.getFloatAttr(f32Type, f32Float);
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, f32Type, value);
        return mlir::LogicalResult::success();
      } else if (mlir::isa<mlir::cir::DoubleType>(op.getType())) {
        auto f64Type = mlir::Float64Type::get(rewriter.getContext());
        // Convert APFloat to f64 precision  
        bool losesInfo = false;
        llvm::APFloat f64Float = apFloat;
        f64Float.convert(llvm::APFloat::IEEEdouble(), llvm::RoundingMode::NearestTiesToEven, &losesInfo);
        value = rewriter.getFloatAttr(f64Type, f64Float);
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, f64Type, value);
        return mlir::LogicalResult::success();
      } else {
        // Fallback to type converter for other floating-point types
        auto convertedTy = getTypeConverter()->convertType(op.getType());
        if (!convertedTy || !mlir::isa<mlir::FloatType>(convertedTy)) {
          return mlir::LogicalResult::failure();
        }
        value = rewriter.getFloatAttr(convertedTy, apFloat);
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, convertedTy, value);
        return mlir::LogicalResult::success();
      }
    } else {
      return mlir::LogicalResult::failure();
    }
    
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, ty, value);
    return mlir::LogicalResult::success();
  }
};

class CIRFuncOpLowering : public mlir::OpConversionPattern<mlir::cir::FuncOp> {
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

struct CIRBrCondOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BrCondOp> {
  using mlir::OpConversionPattern<mlir::cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::arith::TruncIOp>(
        brOp.getLoc(), rewriter.getI1Type(), condition);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        brOp, i1Condition.getResult(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRTernaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::TernaryOp> {
public:
  using OpConversionPattern<mlir::cir::TernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto condition = adaptor.getCond();
    
    // Convert condition to i1 if needed (type converter will handle the conversion)
    auto i1Type = rewriter.getI1Type();
    auto i1Condition = getTypeConverter()->materializeSourceConversion(
        rewriter, op.getLoc(), i1Type, condition);
    if (!i1Condition) {
      return mlir::failure();
    }
    
    SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return mlir::failure();

    auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), resultTypes,
                                                 i1Condition, true);
    auto *thenBlock = &ifOp.getThenRegion().front();
    auto *elseBlock = &ifOp.getElseRegion().front();
    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), thenBlock,
                               thenBlock->end());
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), elseBlock,
                               elseBlock->end());

    rewriter.replaceOp(op, ifOp);
    return mlir::success();
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

    // For yields in scf.if, convert to scf.yield
    if (op->getParentOp() && isa<mlir::scf::IfOp>(op->getParentOp())) {
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getOperands());
      return mlir::LogicalResult::success();
    }

    // Otherwise, convert to scf.yield
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRLoopOpInterfaceLowering
    : public mlir::OpInterfaceConversionPattern<mlir::cir::LoopOpInterface> {
public:
  using mlir::OpInterfaceConversionPattern<
      mlir::cir::LoopOpInterface>::OpInterfaceConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOpInterface op,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {

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

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering, CIRBrOpLowering>(patterns.getContext());

  patterns.add<CIRCmpOpLowering, CIRCallOpLowering, CIRUnaryOpLowering,
               CIRBinOpLowering, CIRLoadOpLowering, CIRConstantOpLowering,
               CIRStoreOpLowering, CIRAllocaOpLowering, CIRFuncOpLowering,
               CIRBrCondOpLowering, CIRTernaryOpLowering,
               CIRYieldOpLowering, CIRLoopOpInterfaceLowering, CIRCosOpLowering>(converter, patterns.getContext());
}

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](mlir::cir::PointerType type) -> mlir::Type {
    auto ty = converter.convertType(type.getPointee());
    // FIXME: The pointee type might not be converted (e.g. struct)
    if (!ty)
      return nullptr;
    return mlir::MemRefType::get({}, ty);
  });
  converter.addConversion(
      [&](mlir::IntegerType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::cir::VoidType type) -> mlir::Type { return {}; });
  converter.addConversion([&](mlir::cir::IntType type) -> mlir::Type {
    // arith dialect ops doesn't take signed integer -- drop cir sign here
    return mlir::IntegerType::get(
        type.getContext(), type.getWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  });
  converter.addConversion([&](mlir::cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8);
  });
  converter.addConversion([&](mlir::cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](mlir::cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](mlir::cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](mlir::cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](mlir::cir::ArrayType type) -> mlir::Type {
    auto elementType = converter.convertType(type.getEltType());
    if (!elementType)
      return nullptr;
    return mlir::MemRefType::get(type.getSize(), elementType);
  });
  
  
  return converter;
}
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
    if (op.getNumResults() == 0) {
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
    addConversion([&](mlir::cir::SingleType type) -> std::optional<mlir::Type> {
      return mlir::Float32Type::get(type.getContext());
    });
    addConversion([&](mlir::cir::DoubleType type) -> std::optional<mlir::Type> {
      return mlir::Float64Type::get(type.getContext());
    });
    
    // Add materialization for i1 <-> i8 conversions (for cir.bool handling)
    addSourceMaterialization([&](mlir::OpBuilder &builder,
                                  mlir::Type resultType, mlir::ValueRange inputs,
                                  mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return nullptr;
      
      auto input = inputs[0];
      auto inputType = input.getType();
      
      // Handle i1 -> i8 (from comparison to bool)
      if (inputType.isInteger(1) && resultType.isInteger(8)) {
        return builder.create<mlir::arith::ExtUIOp>(loc, resultType, input);
      }
      // Handle i8 -> i1 (from bool to condition)
      if (inputType.isInteger(8) && resultType.isInteger(1)) {
        return builder.create<mlir::arith::TruncIOp>(loc, resultType, input);
      }
      
      return nullptr;
    });
    
    addTargetMaterialization([&](mlir::OpBuilder &builder,
                                  mlir::Type resultType, mlir::ValueRange inputs,
                                  mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return nullptr;
      
      auto input = inputs[0];
      auto inputType = input.getType();
      
      // Handle i1 -> i8 (from comparison to bool)
      if (inputType.isInteger(1) && resultType.isInteger(8)) {
        return builder.create<mlir::arith::ExtUIOp>(loc, resultType, input);
      }
      // Handle i8 -> i1 (from bool to condition)
      if (inputType.isInteger(8) && resultType.isInteger(1)) {
        return builder.create<mlir::arith::TruncIOp>(loc, resultType, input);
      }
      
      return nullptr;
    });
    addConversion([&](mlir::cir::ArrayType type) -> std::optional<mlir::Type> {
      auto elementType = convertType(type.getEltType());
      if (!elementType)
        return std::nullopt;
      return mlir::MemRefType::get({static_cast<int64_t>(type.getSize())}, elementType);
    });
    // Add CIR float type conversions
    addConversion([](mlir::cir::SingleType type) -> std::optional<mlir::Type> {
      return mlir::Float32Type::get(type.getContext());
    });
    addConversion([](mlir::cir::DoubleType type) -> std::optional<mlir::Type> {
      return mlir::Float64Type::get(type.getContext());
    });
    addConversion([&](mlir::cir::FP80Type type) -> std::optional<mlir::Type> {
      return mlir::Float80Type::get(type.getContext());
    });
    addConversion([&](mlir::cir::LongDoubleType type) -> std::optional<mlir::Type> {
      return convertType(type.getUnderlying());
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
                         mlir::cf::ControlFlowDialect, mlir::scf::SCFDialect,
                         mlir::math::MathDialect>();
  target.addIllegalDialect<mlir::cir::CIRDialect>();

  mlir::RewritePatternSet patterns(context);
  patterns.add<CIRReturnLowering, CIRFuncOpLowering, CIRCallOpLowering,
               CIRAllocaOpLowering, CIRLoadOpLowering, CIRStoreOpLowering,
               CIRConstantOpLowering, CIRUnaryOpLowering, CIRBinOpLowering,
               CIRCmpOpLowering, CIRBrOpLowering, CIRBrCondOpLowering,
               CIRTernaryOpLowering, CIRYieldOpLowering, CIRIfOpLowering,
               CIRLoopOpInterfaceLowering, CIRScopeOpLowering, CIRCastOpLowering,
               CIRGlobalOpLowering, CIRGetGlobalOpLowering, CIRCosOpLowering>(converter, context);

  if (mlir::failed(
          mlir::applyPartialConversion(theModule, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<llvm::Module>
lowerFromCIRToMLIRToLLVMIR(mlir::ModuleOp theModule,
                           std::unique_ptr<mlir::MLIRContext> mlirCtx,
                           LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createConvertCIRToMLIRPass());
  pm.addPass(createConvertMLIRToLLVMPass());

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

  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
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

} // namespace cir
