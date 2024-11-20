//====- CIRPasses.cpp - Lowering from CIR to LLVM -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements machinery for any CIR <-> CIR passes used by clang.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

namespace cir {
mlir::LogicalResult runCIRToCIRPasses(
    mlir::ModuleOp theModule, mlir::MLIRContext *mlirCtx,
    clang::ASTContext &astCtx, bool enableVerifier, bool enableLifetime,
    llvm::StringRef lifetimeOpts, bool enableIdiomRecognizer,
    llvm::StringRef idiomRecognizerOpts, bool enableLibOpt,
    llvm::StringRef libOptOpts, std::string &passOptParsingFailure,
    bool flattenCIR, bool emitMLIR, bool enableCallConvLowering,
    bool enableMem2Reg) {

  mlir::PassManager pm(mlirCtx);
  pm.addPass(mlir::createMergeCleanupsPass());

  // TODO(CIR): Make this actually propagate errors correctly. This is stubbed
  // in to get rebases going.
  auto errorHandler = [](const llvm::Twine &) -> mlir::LogicalResult {
    return mlir::LogicalResult::failure();
  };

  if (enableLifetime) {
    auto lifetimePass = mlir::createLifetimeCheckPass(&astCtx);
    if (lifetimePass->initializeOptions(lifetimeOpts, errorHandler).failed()) {
      passOptParsingFailure = lifetimeOpts;
      return mlir::failure();
    }
    pm.addPass(std::move(lifetimePass));
  }

  if (enableIdiomRecognizer) {
    auto idiomPass = mlir::createIdiomRecognizerPass(&astCtx);
    if (idiomPass->initializeOptions(idiomRecognizerOpts, errorHandler)
            .failed()) {
      passOptParsingFailure = idiomRecognizerOpts;
      return mlir::failure();
    }
    pm.addPass(std::move(idiomPass));
  }

  if (enableLibOpt) {
    auto libOpPass = mlir::createLibOptPass(&astCtx);
    if (libOpPass->initializeOptions(libOptOpts, errorHandler).failed()) {
      passOptParsingFailure = libOptOpts;
      return mlir::failure();
    }
    pm.addPass(std::move(libOpPass));
  }

  pm.addPass(mlir::createLoweringPreparePass(&astCtx));

  // FIXME(cir): This pass should run by default, but it is lacking support for
  // several code bits. Once it's more mature, we should fix this.
  if (enableCallConvLowering)
    pm.addPass(mlir::createCallConvLoweringPass());

  if (flattenCIR || enableMem2Reg)
    mlir::populateCIRPreLoweringPasses(pm);

  if (enableMem2Reg)
    pm.addPass(mlir::createMem2Reg());

  if (emitMLIR)
    pm.addPass(mlir::createSCFPreparePass());

  // FIXME: once CIRCodenAction fixes emission other than CIR we
  // need to run this right before dialect emission.
  pm.addPass(mlir::createDropASTPass());
  pm.enableVerifier(enableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);
  return pm.run(theModule);
}

} // namespace cir

namespace mlir {

void populateCIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createFlattenCFGPass());
  pm.addPass(createGotoSolverPass());
}

} // namespace mlir
