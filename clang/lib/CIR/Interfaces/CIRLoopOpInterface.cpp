//===- CIRLoopOpInterface.cpp - Interface for CIR loop-like ops *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.cpp.inc"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

void LoopOpInterface::getLoopOpSuccessorRegions(
    LoopOpInterface op, RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // Branching from the parent goes to the entry region (condition for
  // while-style loops, body for do-while).
  // Branching to first region: go to condition or body (do-while).
  if (point.isParent()) {
    regions.emplace_back(&op.getEntry(), op.getEntry().getArguments());
    return;
  }

  auto *terminator = point.getTerminatorPredecessorOrNull();
  assert(terminator && "non-parent branch point must have a terminator");
  Region *stepRegion = op.maybeGetStep();
  Region *branchRegion = terminator->getParentRegion();
  // Branching from condition: go to body or exit.
  if (&op.getCond() == branchRegion) {
    regions.emplace_back(op.getOperation(), op->getResults());
    regions.emplace_back(&op.getBody(), op.getBody().getArguments());
    return;
  }
  // Branching from body: go to step (for) or condition.
  if (&op.getBody() == branchRegion) {
    // FIXME(cir): Should we consider break/continue statements here?
    Region *afterBody = stepRegion;
    if (!afterBody)
      afterBody = &op.getCond();
    regions.emplace_back(afterBody, afterBody->getArguments());
    return;
  }
  // Branching from step: go to condition.
  if (!stepRegion && op.getOperation()->getNumRegions() > 2)
    stepRegion = &op.getOperation()->getRegion(2);
  if (stepRegion == branchRegion) {
    regions.emplace_back(&op.getCond(), op.getCond().getArguments());
    return;
  }
  llvm_unreachable("unexpected loop branch point");
}

/// Verify invariants of the LoopOpInterface.
LogicalResult detail::verifyLoopOpInterface(Operation *op) {
  auto loopOp = cast<LoopOpInterface>(op);
  if (!isa<ConditionOp>(loopOp.getCond().back().getTerminator()))
    return op->emitOpError(
        "expected condition region to terminate with 'cir.condition'");
  return success();
}

} // namespace cir
} // namespace mlir
