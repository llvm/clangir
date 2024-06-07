#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include <iostream>

using namespace mlir;
using namespace mlir::cir;

namespace {


struct CIRSimplifyPass : public CIRSimplifyBase<CIRSimplifyPass> {

  CIRSimplifyPass() = default;
  void runOnOperation() override;
};

class SimplifyDoWhileZero
    : public mlir::OpRewritePattern<mlir::cir::DoWhileOp> {

  bool isZero(mlir::Value v) const {
    if (auto c = dyn_cast<mlir::cir::CastOp>(v.getDefiningOp()))
      return isZero(c.getSrc());

    auto c = dyn_cast<mlir::cir::ConstantOp>(v.getDefiningOp());
    return c && c.isIntZero();
  }

  void replaceWithBranch(mlir::Operation *op, mlir::Block *dest,
                         mlir::PatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);    
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<mlir::cir::BrOp>(op, dest);
  }
  
  bool isSingleBlockScope(Operation* op) const {
    auto scope = dyn_cast<mlir::cir::ScopeOp>(op);
    return scope && scope.getScopeRegion().hasOneBlock();
  }

public:
  using mlir::OpRewritePattern<
      mlir::cir::DoWhileOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::DoWhileOp op,
                  mlir::PatternRewriter &rewriter) const final {

    auto& cond = op.getCond().front();
    auto conditionOp = cast<mlir::cir::ConditionOp>(cond.getTerminator());

    if (!isZero(conditionOp.getCondition()) 
        || !isSingleBlockScope(op->getParentOp()))
      return mlir::failure();

    auto scope = dyn_cast<mlir::cir::ScopeOp>(op->getParentOp());      
    mlir::Block& scopeBody = scope.getScopeRegion().front();   

    auto *entry = rewriter.getInsertionBlock();
    auto term = scopeBody.getTerminator();
    auto* exit = scopeBody.splitBlock(term);
    
    rewriter.setInsertionPointToEnd(entry);
    rewriter.create<mlir::cir::BrOp>(op.getLoc(), &op.getEntry().front());

    for (auto &b : op.getBody().getBlocks()) {
      auto op = b.getTerminator();
      if (isa<mlir::cir::BreakOp, 
              mlir::cir::ContinueOp, 
              mlir::cir::YieldOp>(op))
        replaceWithBranch(op, exit, rewriter);
    }

    rewriter.inlineRegionBefore(op.getBody(), exit);
    rewriter.eraseOp(op);
   
    return mlir::success();
  }
};

void CIRSimplifyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<SimplifyDoWhileZero>(patterns.getContext());

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<DoWhileOp>(op))
      ops.push_back(op);
  });

  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed()) 
    signalPassFailure();

}
} // namespace

std::unique_ptr<Pass> mlir::createCIRSimplifyPass() {
  return std::make_unique<CIRSimplifyPass>();
}