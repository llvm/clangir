#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;


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


struct RegionInliningPass : public RegionInliningBase<RegionInliningPass> {
  using RegionInliningBase::RegionInliningBase;

  void runOnOperation() override;
  void process_gotos(SmallVector<GotoOp>& gotos);
};

struct IfOpRegionsInlining : public OpRewritePattern<IfOp> {
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
                                         thenBeforeBody,
                                         elseBeforeBody);

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

struct ScopeOpRegionsInlining : public OpRewritePattern<ScopeOp> {
  using OpRewritePattern<ScopeOp>::OpRewritePattern;

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

class LoopOpRegionsInlining
    : public mlir::OpInterfaceRewritePattern<mlir::cir::LoopOpInterface> {
public:
  using mlir::OpInterfaceRewritePattern<
      mlir::cir::LoopOpInterface>::OpInterfaceRewritePattern;

  inline void
  lowerConditionOp(mlir::cir::ConditionOp op, mlir::Block *body,
                   mlir::Block *exit,
                   mlir::PatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::cir::BrCondOp>(op, op.getCondition(),
                                                     body, exit);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoopOpInterface op,
                  mlir::PatternRewriter &rewriter) const override {
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
    // auto bodyYield = dyn_cast<mlir::cir::YieldOp>(body->getTerminator());
    // if (bodyYield)
    //   lowerTerminator(bodyYield, (step ? step : cond), rewriter);

    for (auto& blk : op.getBody().getBlocks()) {
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



void populateRegionInliningPatterns(RewritePatternSet &patterns) { 
  // clang-format off
  patterns.add<
    IfOpRegionsInlining,
    ScopeOpRegionsInlining,
    LoopOpRegionsInlining
  >(patterns.getContext());
  // clang-format on
}

// TODO: this is still a draft, not sure it's a good way. Maybe create a separate
// pass for it
void RegionInliningPass::process_gotos(SmallVector<GotoOp>& gotos) {
  for (auto& goTo : gotos) {
    mlir::ConversionPatternRewriter rewriter(goTo.getContext());
    rewriter.setInsertionPoint(goTo);
    auto func = goTo.getOperation()->getParentOfType<mlir::cir::FuncOp>();

    func.getBody().walk([&](mlir::Block* blk) {
      for (auto& op : blk->getOperations()) {
        if (auto lab = dyn_cast<mlir::cir::LabelOp>(op)) {
          if (goTo.getLabel() == lab.getLabel()){
            lowerTerminator(goTo, blk, rewriter);
            goTo.erase();
            return mlir::WalkResult::interrupt();
          }
        }
      }
      return mlir::WalkResult::advance();
    });
  }
}

void RegionInliningPass::runOnOperation() {

  RewritePatternSet patterns(&getContext());
  populateRegionInliningPatterns(patterns);

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  SmallVector<GotoOp> gotos;
  getOperation()->walk([&](Operation *op) {
    if (isa<IfOp, LoopOpInterface, ScopeOp>(op))
      ops.push_back(op);

    if (auto goTo = dyn_cast<GotoOp>(op))
        gotos.push_back(goTo);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();

  process_gotos(gotos);
    getOperation()->dump();


}

} // namespace

std::unique_ptr<Pass> mlir::createRegionInliningPass() {
  return std::make_unique<RegionInliningPass>();
}