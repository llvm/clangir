//===- MLIRCoreDialectsLoweringPrepare.cpp - CIR lowering preparation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace llvm;
using namespace cir;

namespace cir {

struct MLIRLoweringPrepare
    : public mlir::PassWrapper<MLIRLoweringPrepare,
                               mlir::OperationPass<mlir::ModuleOp>> {
  // `scf.index_switch` requires that switch branches do not fall through.
  // We need to copy the next branch's body when the current `cir.case` does
  // not terminate with a break.
  void removeFallthrough(llvm::SmallVector<CaseOp> &cases);

  void runOnOp(mlir::Operation *op);
  void runOnOperation() final;

  StringRef getDescription() const override {
    return "Rewrite CIR module to be more 'scf' dialect-friendly";
  }

  StringRef getArgument() const override { return "mlir-lowering-prepare"; }
};

// `scf.index_switch` requires that switch branches do not fall through.
// We need to copy the next branch's body when the current `cir.case` does not
// terminate with a break.
void MLIRLoweringPrepare::removeFallthrough(llvm::SmallVector<CaseOp> &cases) {
  CIRBaseBuilderTy builder(getContext());
  // Note we enumerate in the reverse order, to facilitate the cloning.
  for (auto it = cases.rbegin(); it != cases.rend(); it++) {
    auto caseOp = *it;
    auto &region = caseOp.getRegion();
    auto &lastBlock = region.back();
    mlir::Operation &last = lastBlock.back();
    if (isa<BreakOp>(last))
      continue;

    // The last op must be a `cir.yield`. As it falls through, we copy the
    // previous case's body to this one.
    if (!isa<YieldOp>(last)) {
      caseOp->dump();
      continue;
    }
    assert(isa<YieldOp>(last));

    // If there's no previous case, we can simply change the yield into a break.
    if (it == cases.rbegin()) {
      builder.setInsertionPointAfter(&last);
      builder.create<BreakOp>(last.getLoc());
      last.erase();
      continue;
    }

    auto prevIt = it;
    --prevIt;
    CaseOp &prev = *prevIt;
    auto &prevRegion = prev.getRegion();
    mlir::IRMapping mapping;
    builder.cloneRegionBefore(prevRegion, region, region.end());

    // We inline the block to the end.
    // This is required because `scf.index_switch` expects that each of its
    // region contains a single block.
    mlir::Block *cloned = lastBlock.getNextNode();
    for (auto it = cloned->begin(); it != cloned->end();) {
      auto next = it;
      next++;
      it->moveBefore(&last);
      it = next;
    }
    cloned->erase();
    last.erase();
  }
}

void MLIRLoweringPrepare::runOnOp(mlir::Operation *op) {
  if (auto switchOp = dyn_cast<SwitchOp>(op)) {
    llvm::SmallVector<CaseOp> cases;
    if (!switchOp.isSimpleForm(cases))
      op->emitError("NYI");

    removeFallthrough(cases);
    return;
  }
  op->emitError("unexpected op type");
}

void MLIRLoweringPrepare::runOnOperation() {
  auto module = getOperation();

  llvm::SmallVector<mlir::Operation *> opsToTransform;
  module->walk([&](mlir::Operation *op) {
    if (isa<SwitchOp>(op))
      opsToTransform.push_back(op);
  });

  for (auto *op : opsToTransform)
    runOnOp(op);
}

std::unique_ptr<mlir::Pass> createMLIRCoreDialectsLoweringPreparePass() {
  return std::make_unique<MLIRLoweringPrepare>();
}

} // namespace cir
