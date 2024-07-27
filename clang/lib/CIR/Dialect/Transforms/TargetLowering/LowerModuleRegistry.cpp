//===- LowerModuleRegistry.cpp - LowerModule singleton registry -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the registry of LowerModule so that it can be easily
// accessed from other libraries.
//
//===----------------------------------------------------------------------===//

// FIXME(cir): This header file is not exposed to the public API, but can be
// reused by CIR ABI lowering since it holds target-specific information.
#include "clang/CIR/Dialect/Transforms/TargetLowering/LowerModuleRegistry.h"
#include "../../../../Basic/Targets.h"
#include "LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace cir {

void LowerModuleRegistry::initializeWithModule(ModuleOp module) {
  assert(!isInitialized() && "LowerModuleRegistry already initialized");
  // Create a new rewriter.
  rewriter.emplace(module->getContext());
  // Fetch the LLVM data layout string.
  auto dataLayoutStr = cast<StringAttr>(
      module->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));

  // Fetch target information.
  llvm::Triple triple(
      cast<StringAttr>(module->getAttr("cir.triple")).getValue());
  clang::TargetOptions targetOptions;
  targetOptions.Triple = triple.str();
  auto targetInfo = clang::targets::AllocateTarget(triple, targetOptions);

  // FIXME(cir): This just uses the default language options. We need to account
  // for custom options.
  // Create context.
  assert(!::cir::MissingFeatures::langOpts());
  clang::LangOptions langOpts;

  lowerModule = std::make_unique<LowerModule>(langOpts, module, dataLayoutStr,
                                              std::move(targetInfo), *rewriter);
}

LowerModuleRegistry &LowerModuleRegistry::instance() {
  static llvm::ManagedStatic<LowerModuleRegistry> lowerModuleRegistry;
  return *lowerModuleRegistry;
}

} // namespace cir
} // namespace mlir
