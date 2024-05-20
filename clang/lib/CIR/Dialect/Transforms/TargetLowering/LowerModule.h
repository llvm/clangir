//===--- LowerModule.h - Abstracts CIR's module lowering --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenModule.h. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H

#include "CIRContext.h"
#include "LowerTypes.h"
#include "MissingFeature.h"
#include "TargetLoweringInfo.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {
namespace cir {

/// Replaces CodeGenModule from Clang in ABI lowering.
class LowerModule {
  // FIXME(cir): This abstraction is not very useful withing CIR's lowering
  // context. I'm keeping it here for parity, but it should probably be removed.

private:
  CIRContext &context;
  ModuleOp module;
  const clang::TargetInfo &Target;
  mutable std::unique_ptr<TargetLoweringInfo> TheTargetCodeGenInfo;
  std::unique_ptr<CIRCXXABI> ABI;

  LowerTypes types;

  PatternRewriter &rewriter;

public:
  LowerModule(CIRContext &C, ModuleOp &module, StringAttr DL,
              const clang::TargetInfo &target, PatternRewriter &rewriter);
  ~LowerModule() = default;

  // Trivial getters.
  LowerTypes &getTypes() { return types; }
  CIRContext &getContext() { return context; }
  CIRCXXABI &getCXXABI() const { return *ABI; }
  const clang::TargetInfo &getTarget() const { return Target; }
  MLIRContext *getMLIRContext() { return module.getContext(); }
  ModuleOp &getModule() { return module; }

  const TargetLoweringInfo &getTargetLoweringInfo();

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  const clang::TargetInfo &getTargetInfo() const { return Target; }

  // FIXME(cir): This would be in ASTContext, not CodeGenModule.
  clang::TargetCXXABI::Kind getCXXABIKind() const {
    auto kind = getTarget().getCXXABI().getKind();
    assert(MissingFeature::langOpts());
    return kind;
  }

  // Rewrite CIR FuncOp to match the target ABI.
  LogicalResult rewriteGlobalFunctionDefinition(FuncOp op, LowerModule &state);

  // Rewrite CIR CallOp to match the target ABI.
  LogicalResult rewriteFunctionCall(CallOp caller, FuncOp callee);
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H
