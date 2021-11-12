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
    auto ty = mlir::MemRefType::get({}, op.type());
    rewriter.replaceOpWithNewOp<mlir::memref::AllocOp>(op, ty);
    return mlir::LogicalResult::success();
  }
};

class CIRLoadLowering : public mlir::OpRewritePattern<mlir::cir::LoadOp> {
public:
  using OpRewritePattern<mlir::cir::LoadOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert(false && "NYI");
    return mlir::LogicalResult::success();
  }
};

class CIRStoreLowering : public mlir::OpRewritePattern<mlir::cir::StoreOp> {
public:
  using OpRewritePattern<mlir::cir::StoreOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert(false && "NYI");
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
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op.getType(),
                                                         op.value());
    return mlir::LogicalResult::success();
  }
};

void populateCIRToMemRefConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<CIRAllocaLowering, CIRLoadLowering, CIRStoreLowering,
               CIRConstantLowering>(patterns.getContext());
}

void populateCIRToStdConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<CIRReturnLowering>(patterns.getContext());
}

void ConvertCIRToLLVMPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  populateCIRToStdConversionPatterns(patterns);
  populateCIRToMemRefConversionPatterns(patterns);
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
  target.addLegalDialect<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
                         mlir::memref::MemRefDialect, mlir::StandardOpsDialect,
                         mlir::cir::CIRDialect>();
  target.addIllegalOp<mlir::cir::AllocaOp, mlir::cir::ConstantOp>();

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
