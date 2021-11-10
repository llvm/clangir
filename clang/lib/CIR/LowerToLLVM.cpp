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
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

namespace cir {

namespace {
struct CIRToLLVMLoweringPass
    : public mlir::PassWrapper<CIRToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::StandardOpsDialect,
                    mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // end anonymous namespace

class CIRReturnLowering : public mlir::OpRewritePattern<mlir::cir::ReturnOp> {
public:
  using OpRewritePattern<mlir::cir::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert(op.getNumOperands() == 0 &&
           "we aren't handling non-zero operand count returns yet");
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
    return mlir::LogicalResult::success();
  }
};

class CIRAllocaLowering : public mlir::OpRewritePattern<mlir::cir::AllocaOp> {
public:
  using OpRewritePattern<mlir::cir::AllocaOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op,
                  mlir::PatternRewriter &rewriter) const override {
    assert(false && "NYI");
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

void populateCIRToStdConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<CIRAllocaLowering, CIRLoadLowering, CIRReturnLowering,
               CIRStoreLowering>(patterns.getContext());
}

void CIRToLLVMLoweringPass::runOnOperation() {
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter typeConverter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  populateCIRToStdConversionPatterns(patterns);
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<llvm::Module>
lowerFromCIRToLLVMIR(mlir::ModuleOp theModule,
                     std::unique_ptr<mlir::MLIRContext> mlirCtx,
                     llvm::LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createLowerToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    llvm::report_fatal_error(
        "The pass manager failed to lower CIR to llvm IR!");

  mlir::registerLLVMDialectTranslation(*mlirCtx);

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    llvm::report_fatal_error("Lowering from llvm dialect to llvm IR failed!");

  return llvmModule;
}

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<CIRToLLVMLoweringPass>();
}

} // namespace cir
