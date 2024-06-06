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

class CIRDoWhileOpSimplification
    : public mlir::OpRewritePattern<mlir::cir::DoWhileOp> {

  bool isZero(mlir::Value v) const {
    if (auto cast = dyn_cast<mlir::cir::CastOp>(v.getDefiningOp()))
      return isZero(cast.getSrc());
    auto c = dyn_cast<mlir::cir::ConstantOp>(v.getDefiningOp());
    return c && c.isIntZero();
  }

public:
  using mlir::OpRewritePattern<
      mlir::cir::DoWhileOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::DoWhileOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto& condReg = op.getCond();
    auto& bodyReg = op.getBody();
    auto *next = rewriter.getInsertionBlock();

    if (!condReg.hasOneBlock())
      return mlir::success();

    auto& condBlk = condReg.front();
    auto term = condBlk.getTerminator();
    if (auto cond = dyn_cast<mlir::cir::ConditionOp>(term)) {
      if (isZero(cond.getCondition())) {
        rewriter.inlineRegionBefore(bodyReg, next);
        rewriter.eraseOp(op);
      }
    }

    return mlir::success();
  }
};


void CIRSimplifyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<CIRDoWhileOpSimplification>(patterns.getContext());

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