//===- CIRSimplify.cpp - performs CIR canonicalization --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

namespace {

/// Removes branches between two blocks if it is the only branch.
///
/// From:
///   ^bb0:
///     cir.br ^bb1
///   ^bb1:  // pred: ^bb0
///     cir.return
///
/// To:
///   ^bb0:
///     cir.return
struct RemoveRedundantBranches : public OpRewritePattern<BrOp> {
  using OpRewritePattern<BrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrOp op,
                                PatternRewriter &rewriter) const final {
    Block *block = op.getOperation()->getBlock();
    Block *dest = op.getDest();

    if (isa<cir::LabelOp>(dest->front()))
      return failure();

    // Single edge between blocks: merge it.
    if (block->getNumSuccessors() == 1 &&
        dest->getSinglePredecessor() == block) {
      rewriter.eraseOp(op);
      rewriter.mergeBlocks(dest, block);
      return success();
    }

    return failure();
  }
};

struct RemoveEmptyScope : public OpRewritePattern<ScopeOp> {
  using OpRewritePattern<ScopeOp>::OpRewritePattern;

  LogicalResult match(ScopeOp op) const final {
    return success(op.getRegion().empty() ||
                   (op.getRegion().getBlocks().size() == 1 &&
                    op.getRegion().front().empty()));
  }

  void rewrite(ScopeOp op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

struct RemoveEmptySwitch : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern<SwitchOp>::OpRewritePattern;

  LogicalResult match(SwitchOp op) const final {
    return success(op.getBody().empty() ||
                   isa<YieldOp>(op.getBody().front().front()));
  }

  void rewrite(SwitchOp op, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
  }
};

struct RemoveTrivialTry : public OpRewritePattern<TryOp> {
  using OpRewritePattern<TryOp>::OpRewritePattern;

  LogicalResult match(TryOp op) const final {
    // FIXME: also check all catch regions are empty
    // return success(op.getTryRegion().hasOneBlock());
    return mlir::failure();
  }

  void rewrite(TryOp op, PatternRewriter &rewriter) const final {
    // Move try body to the parent.
    assert(op.getTryRegion().hasOneBlock());

    Block *parentBlock = op.getOperation()->getBlock();
    mlir::Block *tryBody = &op.getTryRegion().getBlocks().front();
    YieldOp y = dyn_cast<YieldOp>(tryBody->getTerminator());
    assert(y && "expected well wrapped up try block");
    y->erase();

    rewriter.inlineBlockBefore(tryBody, parentBlock, Block::iterator(op));
    rewriter.eraseOp(op);
  }
};

// Remove call exception with empty cleanups
struct SimplifyCallOp : public OpRewritePattern<CallOp> {
  using OpRewritePattern<CallOp>::OpRewritePattern;

  LogicalResult match(CallOp op) const final {
    // Applicable to cir.call exception ... clean { cir.yield }
    mlir::Region *r = &op.getCleanup();
    if (r->empty() || !r->hasOneBlock())
      return failure();

    mlir::Block *b = &r->getBlocks().back();
    if (&b->back() != &b->front())
      return failure();

    return success(isa<YieldOp>(&b->getOperations().back()));
  }

  void rewrite(CallOp op, PatternRewriter &rewriter) const final {
    mlir::Block *b = &op.getCleanup().back();
    rewriter.eraseOp(&b->back());
    rewriter.eraseBlock(b);
  }
};

//===----------------------------------------------------------------------===//
// CIRCanonicalizePass
//===----------------------------------------------------------------------===//

struct CIRCanonicalizePass : public CIRCanonicalizeBase<CIRCanonicalizePass> {
  using CIRCanonicalizeBase::CIRCanonicalizeBase;

  // The same operation rewriting done here could have been performed
  // by CanonicalizerPass (adding hasCanonicalizer for target Ops and
  // implementing the same from above in CIRDialects.cpp). However, it's
  // currently too aggressive for static analysis purposes, since it might
  // remove things where a diagnostic can be generated.
  //
  // FIXME: perhaps we can add one more mode to GreedyRewriteConfig to
  // disable this behavior.
  void runOnOperation() override;
};

void populateCIRCanonicalizePatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    RemoveRedundantBranches,
    RemoveEmptyScope,
    RemoveEmptySwitch,
    RemoveTrivialTry,
    SimplifyCallOp
  >(patterns.getContext());
  // clang-format on
}

void CIRCanonicalizePass::runOnOperation() {
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateCIRCanonicalizePatterns(patterns);

  // Collect operations to apply patterns.
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    // CastOp here is to perform a manual `fold` in
    // applyOpPatternsAndFold
    if (isa<BrOp, BrCondOp, ScopeOp, SwitchOp, CastOp, TryOp, UnaryOp, SelectOp,
            ComplexCreateOp, ComplexRealOp, ComplexImagOp, CallOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRCanonicalizePass() {
  return std::make_unique<CIRCanonicalizePass>();
}
