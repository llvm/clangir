#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"


using namespace mlir;
using namespace mlir::cir;

namespace {

struct StructuredCFGPass : public StructuredCFGBase<StructuredCFGPass> {
  
  StructuredCFGPass() = default;
  void runOnOperation() override;
};

void populateStructuredCFGPatterns(RewritePatternSet &patterns) { 
  //TODO: add patterns here
}

void StructuredCFGPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateStructuredCFGPatterns(patterns);

  // Collect operations to apply patterns.  
  SmallVector<Operation *, 16> ops;
  getOperation()->walk<mlir::WalkOrder::PostOrder>([&](Operation *op) {
     //TODO: push back operations here
  });
  
  // Apply patterns.
  if (applyOpPatternsAndFold(ops, std::move(patterns)).failed())
    signalPassFailure();
}

} // namespace


namespace mlir {

std::unique_ptr<Pass> createStructuredCFGPass() {
  return std::make_unique<StructuredCFGPass>();
}

}
