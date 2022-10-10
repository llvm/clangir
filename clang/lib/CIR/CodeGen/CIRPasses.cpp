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

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace cir {
mlir::LogicalResult runCIRToCIRPasses(mlir::ModuleOp theModule,
                                      mlir::MLIRContext *mlirCtx,
                                      bool enableVerifier, bool enableLifetime,
                                      llvm::StringRef lifetimeOpts,
                                      bool &passOptParsingFailure) {
  mlir::PassManager pm(mlirCtx);
  passOptParsingFailure = false;

  pm.addPass(mlir::createMergeCleanupsPass());

  if (enableLifetime) {
    auto lifetimePass = mlir::createLifetimeCheckPass();
    if (lifetimePass->initializeOptions(lifetimeOpts).failed()) {
      passOptParsingFailure = true;
      return mlir::failure();
    }
    pm.addPass(std::move(lifetimePass));
  }

  pm.enableVerifier(enableVerifier);

  return pm.run(theModule);
}
} // namespace cir
