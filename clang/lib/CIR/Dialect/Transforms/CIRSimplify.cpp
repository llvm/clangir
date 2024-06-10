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

//TODO: there is something bad here 
static bool has_one_user(mlir::Value val) {
  unsigned users = 0;      
  for (auto it = val.user_begin(); it != val.user_end(); ++it)  
    users++;            
  return users == 1;  
}

struct CIRSimplifyPass : public CIRSimplifyBase<CIRSimplifyPass> {

  CIRSimplifyPass() = default;
  void runOnOperation() override;
};

class SimplifyCastOp
    : public mlir::OpRewritePattern<mlir::cir::CastOp> {

  bool isOpposite(mlir::cir::CastKind kind1, mlir::cir::CastKind kind2) const {
    return kind1 == mlir::cir::CastKind::bool_to_int && kind2 == mlir::cir::CastKind::int_to_bool;
  }

public:
  using mlir::OpRewritePattern<mlir::cir::CastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp op,
                  mlir::PatternRewriter &rewriter) const final {
      auto val = op.getResult();
      if (has_one_user(val)) {    
        auto user = *val.user_begin();
        if (auto cast = dyn_cast<mlir::cir::CastOp>(user)) {
          if (isOpposite(op.getKind(), cast.getKind())) {
            auto def = op.getSrc();
            rewriter.replaceAllUsesWith(cast, def);
            rewriter.eraseOp(cast);
            rewriter.eraseOp(op);
          }
        }      
      }
      
      return mlir::success();
  }
};


class SimplifyUnaryNot : public mlir::OpRewritePattern<mlir::cir::UnaryOp> {

  bool is_not(mlir::cir::UnaryOp op) const {
    return op.getKind() == mlir::cir::UnaryOpKind::Not;
  } 

public:
  using mlir::OpRewritePattern<mlir::cir::UnaryOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (!is_not(op) || !has_one_user(op.getResult()))
      return mlir::success();
    
    auto val = op.getResult();    
    auto user = *val.user_begin();
    if (auto unop = dyn_cast<mlir::cir::UnaryOp>(user)) {
      if (!is_not(unop))
        return mlir::success();               
          rewriter.replaceAllUsesWith(unop, op.getInput());
          rewriter.eraseOp(unop);
          rewriter.eraseOp(op);
    }

    return mlir::success();

  }
};

void CIRSimplifyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<SimplifyCastOp, SimplifyUnaryNot>(patterns.getContext());

  // Collect operations to apply patterns.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<mlir::cir::CastOp, mlir::cir::UnaryOp>(op))
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