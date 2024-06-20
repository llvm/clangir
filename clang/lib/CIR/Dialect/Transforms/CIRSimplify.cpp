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

struct CIRSimplifyPass : public CIRSimplifyBase<CIRSimplifyPass> {
  CIRSimplifyPass() = default;
  void runOnOperation() override;
};

class SimplifyBoolCasts : public mlir::OpRewritePattern<mlir::cir::CastOp> {

  typedef std::vector<mlir::cir::CastOp> chain_of_casts;

  bool is_bool_aimed(mlir::cir::CastOp op) const {
    return op.getKind() == mlir::cir::CastKind::int_to_bool;
  }

  bool is_int_or_bool_cast(mlir::cir::CastOp op) const {
    auto kind = op.getKind();
    return kind == mlir::cir::CastKind::bool_to_int ||
           kind == mlir::cir::CastKind::int_to_bool ||
           kind == mlir::cir::CastKind::integral;
  }

  // makes a chain of casts with the next properties:
  // - each member is a cast of one of the next kinds: integral, bool_to_int,
  // int_to_bool
  // - each member except the last one may have only one use - the next member
  // in the chain
  // - last member may have one or many users
  void collect_casts(chain_of_casts &casts, mlir::cir::CastOp op) const {
    auto val = op.getResult();
    if (is_int_or_bool_cast(op)) {
      casts.push_back(op);
      if (val.hasOneUse())
        if (auto next = dyn_cast<mlir::cir::CastOp>(*val.user_begin()))
          collect_casts(casts, next);
    }
  }

public:
  using mlir::OpRewritePattern<mlir::cir::CastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp op,
                  mlir::PatternRewriter &rewriter) const final {

    chain_of_casts casts;
    collect_casts(casts, op);

    if (casts.size() < 2 || !is_bool_aimed(casts.back()))
      return mlir::failure();

    auto first = casts.front();
    auto last = casts.back();

    mlir::Value actual;
    auto it = casts.begin();

    // The next two cases are covered:
    // 1) bool_to_int -> ... -> int_to_bool: remove all the casts:
    //    bool value was already obtained before all the conversions
    // 2) int_to_bool -> ... -> int_to_bool: take the result of the first op,
    //    erase all others
    if (first.getKind() == mlir::cir::CastKind::bool_to_int) {
      actual = first.getSrc();
    } else if (first.getKind() == mlir::cir::CastKind::int_to_bool) {
      actual = first.getResult();
      it++;
    } else {
      return mlir::failure();
    }

    rewriter.replaceAllUsesWith(last, actual);

    for (auto rit = casts.end() - 1; rit >= it; --rit)
      rewriter.eraseOp(*rit);

    return mlir::success();
  }
};

class SimplifyUnaryNot : public mlir::OpRewritePattern<mlir::cir::UnaryOp> {

  bool is_unary_not(mlir::cir::UnaryOp op) const {
    return op.getKind() == mlir::cir::UnaryOpKind::Not;
  }

public:
  using mlir::OpRewritePattern<mlir::cir::UnaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (!is_unary_not(op) || !op.getResult().hasOneUse())
      return mlir::failure();

    auto val = op.getResult();
    auto user = *val.user_begin();

    if (auto next = dyn_cast<mlir::cir::UnaryOp>(user)) {
      if (!is_unary_not(next))
        return mlir::failure();
      rewriter.replaceAllUsesWith(next, op.getInput());
      rewriter.eraseOp(next);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

void CIRSimplifyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<SimplifyBoolCasts, SimplifyUnaryNot>(patterns.getContext());

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation *op) {
    if (isa<mlir::cir::UnaryOp, mlir::cir::CastOp>(op))
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