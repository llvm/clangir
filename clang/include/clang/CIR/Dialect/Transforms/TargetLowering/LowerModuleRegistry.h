//===- LowerModuleRegistry.h - LowerModule singleton registry ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the registry of LowerModule so that it can be easily
// accessed from other libraries.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULEREGISTRY_H
#define CLANG_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULEREGISTRY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cir {

class LowerModule;

/// Registry for the LowerModule, enabling easy access to the LowerModule from
/// various libraries.
class LowerModuleRegistry {
  std::unique_ptr<LowerModule> lowerModule;
  std::optional<PatternRewriter> rewriter;

public:
  /// Initialize the LowerModuleRegistry with the given module and an internal
  /// rewriter.
  void initializeWithModule(ModuleOp module);

  /// Check if the LowerModuleRegistry has been initialized.
  bool isInitialized() { return lowerModule != nullptr; }

  /// Get the reference to already-initialized LowerModule.
  LowerModule &get() {
    assert(isInitialized() && "LowerModuleRegistry not initialized");
    return *lowerModule;
  }

  /// Get the LowerModuleRegistry singleton.
  static LowerModuleRegistry &instance();
};

} // namespace cir
} // namespace mlir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULEREGISTRY_H
