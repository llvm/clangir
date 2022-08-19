//====- LowerToLLVM.cpp - Lowering from CIR to LLVM -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements full lowering of CIR operations to LLVMIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/CIR/IR/CIRDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/Sequence.h"

using namespace cir;
using namespace llvm;

namespace cir {

namespace {
struct ConvertCIRToLLVMPass
    : public mlir::PassWrapper<ConvertCIRToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::StandardOpsDialect,
                    mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-llvm"; }
};

struct ConvertCIRToMemRefPass
    : public mlir::PassWrapper<ConvertCIRToMemRefPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect, mlir::StandardOpsDialect,
                    mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-memref"; }
};
} // end anonymous namespace

class CIRReturnLowering : public mlir::OpRewritePattern<mlir::cir::ReturnOp> {
public:
  using OpRewritePattern<mlir::cir::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, op->getResultTypes(),
                                                op->getOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRAllocaLowering : public mlir::OpRewritePattern<mlir::cir::AllocaOp> {
public:
  using OpRewritePattern<mlir::cir::AllocaOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto type = op.type();
    mlir::MemRefType memreftype;

    if (type.isa<mlir::cir::BoolType>()) {
      auto integerType =
          mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::Signless);
      memreftype = mlir::MemRefType::get({}, integerType);
    } else if (type.isa<mlir::cir::ArrayType>()) {
      mlir::cir::ArrayType arraytype = type.dyn_cast<mlir::cir::ArrayType>();
      memreftype =
          mlir::MemRefType::get(arraytype.getSize(), arraytype.getEltType());
    } else if (type.isa<mlir::IntegerType>() || type.isa<mlir::FloatType>()) {
      memreftype = mlir::MemRefType::get({}, op.type());
    } else {
      llvm_unreachable("type to be allocated not supported yet");
    }
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memreftype,
                                                        op.alignmentAttr());
    return mlir::LogicalResult::success();
  }
};

class CIRLoadLowering : public mlir::ConversionPattern {
public:
  CIRLoadLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(mlir::cir::LoadOp::getOperationName(), 1, ctx) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, operands[0]);
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::ConversionPattern {
public:
  CIRStoreLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(mlir::cir::StoreOp::getOperationName(), 1,
                                ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, operands[0],
                                                       operands[1]);
    return mlir::LogicalResult::success();
  }
};

class CIRConstantLowering
    : public mlir::OpRewritePattern<mlir::cir::ConstantOp> {
public:
  using OpRewritePattern<mlir::cir::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getType().isa<mlir::cir::BoolType>()) {
      mlir::Type type =
          mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::Signless);
      mlir::Attribute IntegerAttr;
      if (op.value() == mlir::BoolAttr::get(getContext(), true))
        IntegerAttr = mlir::IntegerAttr::get(type, 1);
      else
        IntegerAttr = mlir::IntegerAttr::get(type, 0);
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, type,
                                                           IntegerAttr);
    } else
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op.getType(),
                                                           op.value());
    return mlir::LogicalResult::success();
  }
};

class CIRBinOpLowering : public mlir::OpRewritePattern<mlir::cir::BinOp> {
public:
  using OpRewritePattern<mlir::cir::BinOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BinOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert((op.lhs().getType() == op.rhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type type = op.rhs().getType();
    assert((type.isa<mlir::IntegerType>() || type.isa<mlir::FloatType>()) &&
           "operand type not supported yet");

    switch (op.kind()) {
    case mlir::cir::BinOpKind::Add:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Sub:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Mul:
      if (type.isa<mlir::IntegerType>())
        rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Div:
      if (type.isa<mlir::IntegerType>()) {
        if (type.isSignedInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(op, op.getType(),
                                                            op.lhs(), op.rhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(op, op.getType(),
                                                            op.lhs(), op.rhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Rem:
      if (type.isa<mlir::IntegerType>()) {
        if (type.isSignedInteger())
          rewriter.replaceOpWithNewOp<mlir::arith::RemSIOp>(op, op.getType(),
                                                            op.lhs(), op.rhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(op, op.getType(),
                                                            op.lhs(), op.rhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(op, op.getType(),
                                                         op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(op, op.getType(),
                                                       op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(op, op.getType(),
                                                      op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, op.getType(),
                                                       op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Shl:
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(op, op.getType(),
                                                       op.lhs(), op.rhs());
      break;
    case mlir::cir::BinOpKind::Shr:
      if (type.isSignedInteger())
        rewriter.replaceOpWithNewOp<mlir::arith::ShRSIOp>(op, op.getType(),
                                                          op.lhs(), op.rhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ShRUIOp>(op, op.getType(),
                                                          op.lhs(), op.rhs());
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpRewritePattern<mlir::cir::CmpOp> {
public:
  using OpRewritePattern<mlir::cir::CmpOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto type = op.lhs().getType();
    auto integerType =
        mlir::IntegerType::get(getContext(), 1, mlir::IntegerType::Signless);

    switch (op.kind()) {
    case mlir::cir::CmpOpKind::gt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (!type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ugt;
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.lhs(), op.rhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGT),
            op.lhs(), op.rhs());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ge: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::uge;
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.lhs(), op.rhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UGE),
            op.lhs(), op.rhs());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::lt: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ult;
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.lhs(), op.rhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULT),
            op.lhs(), op.rhs());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::le: {
      if (type.isa<mlir::IntegerType>()) {
        mlir::arith::CmpIPredicate cmpIType;
        if (type.isSignlessInteger())
          llvm_unreachable("integer type not supported in CIR yet");
        cmpIType = mlir::arith::CmpIPredicate::ule;
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(), cmpIType),
            op.lhs(), op.rhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::ULE),
            op.lhs(), op.rhs());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::eq: {
      if (type.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::eq),
            op.lhs(), op.rhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UEQ),
            op.lhs(), op.rhs());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    case mlir::cir::CmpOpKind::ne: {
      if (type.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
            op, integerType,
            mlir::arith::CmpIPredicateAttr::get(getContext(),
                                                mlir::arith::CmpIPredicate::ne),
            op.lhs(), op.rhs());
      } else if (type.isa<mlir::FloatType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
            op, integerType,
            mlir::arith::CmpFPredicateAttr::get(
                getContext(), mlir::arith::CmpFPredicate::UNE),
            op.lhs(), op.rhs());
      } else {
        llvm_unreachable("Unknown Operand Type");
      }
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

void populateCIRToMemRefConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<CIRAllocaLowering, CIRLoadLowering, CIRStoreLowering,
               CIRConstantLowering, CIRReturnLowering, CIRBinOpLowering,
               CIRCmpOpLowering>(patterns.getContext());
}

void ConvertCIRToLLVMPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

void ConvertCIRToMemRefPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  // TODO: Should this be a wholesale conversion? It's a bit ambiguous on
  // whether we should have micro-conversions that do the minimal amount of work
  // or macro conversions that entiirely remove a dialect.
  target.addLegalOp<mlir::ModuleOp, mlir::FuncOp>();
  target
      .addLegalDialect<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
                       mlir::memref::MemRefDialect, mlir::StandardOpsDialect>();
  target
      .addIllegalOp<mlir::cir::BinOp, mlir::cir::ReturnOp, mlir::cir::AllocaOp,
                    mlir::cir::LoadOp, mlir::cir::StoreOp,
                    mlir::cir::ConstantOp, mlir::cir::CmpOp>();

  mlir::RewritePatternSet patterns(&getContext());
  populateCIRToMemRefConversionPatterns(patterns);
  // populateAffineToStdConversionPatterns(patterns);
  // populateLoopToStdConversionPatterns(patterns);

  auto module = getOperation();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<llvm::Module>
lowerFromCIRToLLVMIR(mlir::ModuleOp theModule,
                     std::unique_ptr<mlir::MLIRContext> mlirCtx,
                     LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createConvertCIRToMemRefPass());
  pm.addPass(createConvertCIRToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error("The pass manager failed to lower CIR to llvm IR!");

  mlir::registerLLVMDialectTranslation(*mlirCtx);

  LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from llvm dialect to llvm IR failed!");

  return llvmModule;
}

std::unique_ptr<mlir::Pass> createConvertCIRToLLVMPass() {
  return std::make_unique<ConvertCIRToLLVMPass>();
}

std::unique_ptr<mlir::Pass> createConvertCIRToMemRefPass() {
  return std::make_unique<ConvertCIRToMemRefPass>();
}

} // namespace cir
