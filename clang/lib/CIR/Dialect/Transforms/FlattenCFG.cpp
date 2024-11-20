//====- FlattenCFG.cpp - Flatten CIR CFG ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass that inlines CIR operations regions into the parent
// function region.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace mlir::cir;

namespace {

/// Lowers operations with the terminator trait that have a single successor.
void lowerTerminator(mlir::Operation *op, mlir::Block *dest,
                     mlir::PatternRewriter &rewriter) {
  assert(op->hasTrait<mlir::OpTrait::IsTerminator>() && "not a terminator");
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(op, dest);
}

/// Walks a region while skipping operations of type `Ops`. This ensures the
/// callback is not applied to said operations and its children.
template <typename... Ops>
void walkRegionSkipping(mlir::Region &region,
                        mlir::function_ref<void(mlir::Operation *)> callback) {
  region.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<Ops...>(op))
      return mlir::WalkResult::skip();
    callback(op);
    return mlir::WalkResult::advance();
  });
}

struct FlattenCFGPass : public FlattenCFGBase<FlattenCFGPass> {

  FlattenCFGPass() = default;
  void runOnOperation() override;
};

struct CIRIfFlattening : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = ifOp.getLoc();
    auto emptyElse = ifOp.getElseRegion().empty();

    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (ifOp->getResults().size() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline then region
    auto *thenBeforeBody = &ifOp.getThenRegion().front();
    auto *thenAfterBody = &ifOp.getThenRegion().back();
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), continueBlock);

    rewriter.setInsertionPointToEnd(thenAfterBody);
    if (auto thenYieldOp =
            dyn_cast<mlir::cir::YieldOp>(thenAfterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
          thenYieldOp, thenYieldOp.getArgs(), continueBlock);
    }

    rewriter.setInsertionPointToEnd(continueBlock);

    // Has else region: inline it.
    mlir::Block *elseBeforeBody = nullptr;
    mlir::Block *elseAfterBody = nullptr;
    if (!emptyElse) {
      elseBeforeBody = &ifOp.getElseRegion().front();
      elseAfterBody = &ifOp.getElseRegion().back();
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), thenAfterBody);
    } else {
      elseBeforeBody = elseAfterBody = continueBlock;
    }

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cir::BrCondOp>(loc, ifOp.getCondition(),
                                         thenBeforeBody, elseBeforeBody);

    if (!emptyElse) {
      rewriter.setInsertionPointToEnd(elseAfterBody);
      if (auto elseYieldOp =
              dyn_cast<mlir::cir::YieldOp>(elseAfterBody->getTerminator())) {
        rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
            elseYieldOp, elseYieldOp.getArgs(), continueBlock);
      }
    }

    rewriter.replaceOp(ifOp, continueBlock->getArguments());
    return mlir::success();
  }
};

class CIRScopeOpFlattening : public mlir::OpRewritePattern<mlir::cir::ScopeOp> {
public:
  using OpRewritePattern<mlir::cir::ScopeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ScopeOp scopeOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = scopeOp.getLoc();

    // Empty scope: just remove it.
    if (scopeOp.getRegion().empty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Split the current block before the ScopeOp to create the inlining
    // point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    if (scopeOp.getNumResults() == 0)
      continueBlock = remainingOpsBlock;
    else
      llvm_unreachable("NYI");

    // Inline body region.
    auto *beforeBody = &scopeOp.getRegion().front();
    auto *afterBody = &scopeOp.getRegion().back();
    rewriter.inlineRegionBefore(scopeOp.getRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    // TODO(CIR): stackSaveOp
    // auto stackSaveOp = rewriter.create<mlir::LLVM::StackSaveOp>(
    //     loc, mlir::LLVM::LLVMPointerType::get(
    //              mlir::IntegerType::get(scopeOp.getContext(), 8)));
    rewriter.create<mlir::cir::BrOp>(loc, mlir::ValueRange(), beforeBody);

    // Replace the scopeop return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    if (auto yieldOp =
            dyn_cast<mlir::cir::YieldOp>(afterBody->getTerminator())) {
      rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, yieldOp.getArgs(),
                                                   continueBlock);
    }

    // TODO(cir): stackrestore?

    // Replace the op with values return from the body region.
    rewriter.replaceOp(scopeOp, continueBlock->getArguments());

    return mlir::success();
  }
};

class CIRTryOpFlattening : public mlir::OpRewritePattern<mlir::cir::TryOp> {
public:
  using OpRewritePattern<mlir::cir::TryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TryOp tryOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto loc = tryOp.getLoc();

    // Empty scope: just remove it.
    if (tryOp.getTryRegion().empty()) {
      rewriter.eraseOp(tryOp);
      return mlir::success();
    }

    // Split the current block before the TryOp to create the inlining
    // point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    mlir::Block *continueBlock;
    continueBlock = remainingOpsBlock;

    // Inline body region.
    auto *beforeBody = &tryOp.getTryRegion().front();
    auto *afterBody = &tryOp.getTryRegion().back();
    rewriter.inlineRegionBefore(tryOp.getTryRegion(), continueBlock);

    // Branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<mlir::cir::BrOp>(loc, mlir::ValueRange(), beforeBody);

    // Replace the tryOp return with a branch that jumps out of the body.
    rewriter.setInsertionPointToEnd(afterBody);
    auto yieldOp = cast<mlir::cir::YieldOp>(afterBody->getTerminator());
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, continueBlock);

    // TODO: handle the catch clauses and cir.try_call while here.

    rewriter.eraseOp(tryOp);
    return mlir::success();
  }
};

class CIRLoopOpInterfaceFlattening
    : public mlir::OpInterfaceRewritePattern<mlir::cir::LoopOpInterface> {
public:
  using mlir::OpInterfaceRewritePattern<
      mlir::cir::LoopOpInterface>::OpInterfaceRewritePattern;

  inline void lowerConditionOp(mlir::cir::ConditionOp op, mlir::Block *body,
                               mlir::Block *exit,
                               mlir::PatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::cir::BrCondOp>(op, op.getCondition(),
                                                     body, exit);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOpInterface op,
                  mlir::PatternRewriter &rewriter) const final {
    // Setup CFG blocks.
    auto *entry = rewriter.getInsertionBlock();
    auto *exit = rewriter.splitBlock(entry, rewriter.getInsertionPoint());
    auto *cond = &op.getCond().front();
    auto *body = &op.getBody().front();
    auto *step = (op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr);

    // Setup loop entry branch.
    rewriter.setInsertionPointToEnd(entry);
    rewriter.create<mlir::cir::BrOp>(op.getLoc(), &op.getEntry().front());

    // Branch from condition region to body or exit.
    auto conditionOp = cast<mlir::cir::ConditionOp>(cond->getTerminator());
    lowerConditionOp(conditionOp, body, exit, rewriter);

    // TODO(cir): Remove the walks below. It visits operations unnecessarily,
    // however, to solve this we would likely need a custom DialecConversion
    // driver to customize the order that operations are visited.

    // Lower continue statements.
    mlir::Block *dest = (step ? step : cond);
    op.walkBodySkippingNestedLoops([&](mlir::Operation *op) {
      if (isa<mlir::cir::ContinueOp>(op))
        lowerTerminator(op, dest, rewriter);
    });

    // Lower break statements.
    walkRegionSkipping<mlir::cir::LoopOpInterface, mlir::cir::SwitchOp>(
        op.getBody(), [&](mlir::Operation *op) {
          if (isa<mlir::cir::BreakOp>(op))
            lowerTerminator(op, exit, rewriter);
        });

    // Lower optional body region yield.
    for (auto &blk : op.getBody().getBlocks()) {
      auto bodyYield = dyn_cast<mlir::cir::YieldOp>(blk.getTerminator());
      if (bodyYield)
        lowerTerminator(bodyYield, (step ? step : cond), rewriter);
    }

    // Lower mandatory step region yield.
    if (step)
      lowerTerminator(cast<mlir::cir::YieldOp>(step->getTerminator()), cond,
                      rewriter);

    // Move region contents out of the loop op.
    rewriter.inlineRegionBefore(op.getCond(), exit);
    rewriter.inlineRegionBefore(op.getBody(), exit);
    if (step)
      rewriter.inlineRegionBefore(*op.maybeGetStep(), exit);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRSwitchOpFlattening
    : public mlir::OpRewritePattern<mlir::cir::SwitchOp> {
public:
  using OpRewritePattern<mlir::cir::SwitchOp>::OpRewritePattern;

  inline void rewriteYieldOp(mlir::PatternRewriter &rewriter,
                             mlir::cir::YieldOp yieldOp,
                             mlir::Block *destination) const {
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(yieldOp, yieldOp.getOperands(),
                                                 destination);
  }

  // Return the new defaultDestination block.
  Block *condBrToRangeDestination(mlir::cir::SwitchOp op,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::Block *rangeDestination,
                                  mlir::Block *defaultDestination,
                                  APInt lowerBound, APInt upperBound) const {
    assert(lowerBound.sle(upperBound) && "Invalid range");
    auto resBlock = rewriter.createBlock(defaultDestination);
    auto sIntType = mlir::cir::IntType::get(op.getContext(), 32, true);
    auto uIntType = mlir::cir::IntType::get(op.getContext(), 32, false);

    auto rangeLength = rewriter.create<mlir::cir::ConstantOp>(
        op.getLoc(), sIntType,
        mlir::cir::IntAttr::get(op.getContext(), sIntType,
                                upperBound - lowerBound));

    auto lowerBoundValue = rewriter.create<mlir::cir::ConstantOp>(
        op.getLoc(), sIntType,
        mlir::cir::IntAttr::get(op.getContext(), sIntType, lowerBound));
    auto diffValue = rewriter.create<mlir::cir::BinOp>(
        op.getLoc(), sIntType, mlir::cir::BinOpKind::Sub, op.getCondition(),
        lowerBoundValue);

    // Use unsigned comparison to check if the condition is in the range.
    auto uDiffValue = rewriter.create<mlir::cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, diffValue);
    auto uRangeLength = rewriter.create<mlir::cir::CastOp>(
        op.getLoc(), uIntType, CastKind::integral, rangeLength);

    auto cmpResult = rewriter.create<mlir::cir::CmpOp>(
        op.getLoc(), mlir::cir::BoolType::get(op.getContext()),
        mlir::cir::CmpOpKind::le, uDiffValue, uRangeLength);
    rewriter.create<mlir::cir::BrCondOp>(op.getLoc(), cmpResult,
                                         rangeDestination, defaultDestination);
    return resBlock;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SwitchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Empty switch statement: just erase it.
    if (!op.getCases().has_value() || op.getCases()->empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Create exit block.
    rewriter.setInsertionPointAfter(op);
    auto *exitBlock =
        rewriter.splitBlock(rewriter.getBlock(), rewriter.getInsertionPoint());

    // Allocate required data structures (disconsider default case in
    // vectors).
    llvm::SmallVector<mlir::APInt, 8> caseValues;
    llvm::SmallVector<mlir::Block *, 8> caseDestinations;
    llvm::SmallVector<mlir::ValueRange, 8> caseOperands;

    llvm::SmallVector<std::pair<APInt, APInt>> rangeValues;
    llvm::SmallVector<mlir::Block *> rangeDestinations;
    llvm::SmallVector<mlir::ValueRange> rangeOperands;

    // Initialize default case as optional.
    mlir::Block *defaultDestination = exitBlock;
    mlir::ValueRange defaultOperands = exitBlock->getArguments();

    // Track fallthrough between cases.
    mlir::cir::YieldOp fallthroughYieldOp = nullptr;

    // Digest the case statements values and bodies.
    for (size_t i = 0; i < op.getCases()->size(); ++i) {
      auto &region = op.getRegion(i);
      auto caseAttr = cast<mlir::cir::CaseAttr>(op.getCases()->getValue()[i]);

      // Found default case: save destination and operands.
      switch (caseAttr.getKind().getValue()) {
      case mlir::cir::CaseOpKind::Default:
        defaultDestination = &region.front();
        defaultOperands = region.getArguments();
        break;
      case mlir::cir::CaseOpKind::Range:
        assert(caseAttr.getValue().size() == 2 &&
               "Case range should have 2 case value");
        rangeValues.push_back(
            {cast<mlir::cir::IntAttr>(caseAttr.getValue()[0]).getValue(),
             cast<mlir::cir::IntAttr>(caseAttr.getValue()[1]).getValue()});
        rangeDestinations.push_back(&region.front());
        rangeOperands.push_back(region.getArguments());
        break;
      case mlir::cir::CaseOpKind::Anyof:
      case mlir::cir::CaseOpKind::Equal:
        // AnyOf cases kind can have multiple values, hence the loop below.
        for (auto &value : caseAttr.getValue()) {
          caseValues.push_back(cast<mlir::cir::IntAttr>(value).getValue());
          caseOperands.push_back(region.getArguments());
          caseDestinations.push_back(&region.front());
        }
        break;
      }

      // Previous case is a fallthrough: branch it to this case.
      if (fallthroughYieldOp) {
        rewriteYieldOp(rewriter, fallthroughYieldOp, &region.front());
        fallthroughYieldOp = nullptr;
      }

      for (auto &blk : region.getBlocks()) {
        if (blk.getNumSuccessors())
          continue;

        // Handle switch-case yields.
        if (auto yieldOp = dyn_cast<mlir::cir::YieldOp>(blk.getTerminator()))
          fallthroughYieldOp = yieldOp;
      }

      // Handle break statements.
      walkRegionSkipping<mlir::cir::LoopOpInterface, mlir::cir::SwitchOp>(
          region, [&](mlir::Operation *op) {
            if (isa<mlir::cir::BreakOp>(op))
              lowerTerminator(op, exitBlock, rewriter);
          });

      // Extract region contents before erasing the switch op.
      rewriter.inlineRegionBefore(region, exitBlock);
    }

    // Last case is a fallthrough: branch it to exit.
    if (fallthroughYieldOp) {
      rewriteYieldOp(rewriter, fallthroughYieldOp, exitBlock);
      fallthroughYieldOp = nullptr;
    }

    for (size_t index = 0; index < rangeValues.size(); ++index) {
      auto lowerBound = rangeValues[index].first;
      auto upperBound = rangeValues[index].second;

      // The case range is unreachable, skip it.
      if (lowerBound.sgt(upperBound))
        continue;

      // If range is small, add multiple switch instruction cases.
      // This magical number is from the original CGStmt code.
      constexpr int kSmallRangeThreshold = 64;
      if ((upperBound - lowerBound)
              .ult(llvm::APInt(32, kSmallRangeThreshold))) {
        for (auto iValue = lowerBound; iValue.sle(upperBound); (void)iValue++) {
          caseValues.push_back(iValue);
          caseOperands.push_back(rangeOperands[index]);
          caseDestinations.push_back(rangeDestinations[index]);
        }
        continue;
      }

      defaultDestination =
          condBrToRangeDestination(op, rewriter, rangeDestinations[index],
                                   defaultDestination, lowerBound, upperBound);
      defaultOperands = rangeOperands[index];
    }

    // Set switch op to branch to the newly created blocks.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::cir::SwitchFlatOp>(
        op, op.getCondition(), defaultDestination, defaultOperands, caseValues,
        caseDestinations, caseOperands);

    return mlir::success();
  }
};
class CIRTernaryOpFlattening
    : public mlir::OpRewritePattern<mlir::cir::TernaryOp> {
public:
  using OpRewritePattern<mlir::cir::TernaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TernaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
    SmallVector<mlir::Location, 2> locs;
    // Ternary result is optional, make sure to populate the location only
    // when relevant.
    if (op->getResultTypes().size())
      locs.push_back(loc);
    auto *continueBlock =
        rewriter.createBlock(remainingOpsBlock, op->getResultTypes(), locs);
    rewriter.create<mlir::cir::BrOp>(loc, remainingOpsBlock);

    auto &trueRegion = op.getTrueRegion();
    auto *trueBlock = &trueRegion.front();
    mlir::Operation *trueTerminator = trueRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&trueRegion.back());
    auto trueYieldOp = dyn_cast<mlir::cir::YieldOp>(trueTerminator);

    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        trueYieldOp, trueYieldOp.getArgs(), continueBlock);
    rewriter.inlineRegionBefore(trueRegion, continueBlock);

    auto *falseBlock = continueBlock;
    auto &falseRegion = op.getFalseRegion();

    falseBlock = &falseRegion.front();
    mlir::Operation *falseTerminator = falseRegion.back().getTerminator();
    rewriter.setInsertionPointToEnd(&falseRegion.back());
    auto falseYieldOp = dyn_cast<mlir::cir::YieldOp>(falseTerminator);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(
        falseYieldOp, falseYieldOp.getArgs(), continueBlock);
    rewriter.inlineRegionBefore(falseRegion, continueBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<mlir::cir::BrCondOp>(loc, op.getCond(), trueBlock,
                                         falseBlock);

    rewriter.replaceOp(op, continueBlock->getArguments());

    // Ok, we're done!
    return mlir::success();
  }
};

void populateFlattenCFGPatterns(RewritePatternSet &patterns) {
  patterns
      .add<CIRIfFlattening, CIRLoopOpInterfaceFlattening, CIRScopeOpFlattening,
           CIRSwitchOpFlattening, CIRTernaryOpFlattening, CIRTryOpFlattening>(
          patterns.getContext());
}

void FlattenCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateFlattenCFGPatterns(patterns);

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, ScopeOp, SwitchOp, LoopOpInterface, TernaryOp, TryOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace

namespace mlir {

std::unique_ptr<Pass> createFlattenCFGPass() {
  return std::make_unique<FlattenCFGPass>();
}

} // namespace mlir
