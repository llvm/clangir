#include "PassDetail.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TimeProfiler.h"

using namespace mlir;
using namespace cir;

namespace {

struct GotoSolverPass : public GotoSolverBase<GotoSolverPass> {

  GotoSolverPass() = default;
  void runOnOperation() override;
};

static void process(cir::FuncOp func) {

  mlir::OpBuilder rewriter(func.getContext());
  llvm::StringMap<Block *> labels;
  llvm::SmallVector<cir::GotoOp, 4> gotos;
  llvm::SmallSet<StringRef, 4> blockAddrLabel;

  func.getBody().walk([&](mlir::Operation *op) {
    if (auto lab = dyn_cast<cir::LabelOp>(op)) {
      labels.try_emplace(lab.getLabel(), lab->getBlock());
    } else if (auto goTo = dyn_cast<cir::GotoOp>(op)) {
      gotos.push_back(goTo);
    } else if (auto blockAddr = dyn_cast<cir::BlockAddressOp>(op)) {
      blockAddrLabel.insert(blockAddr.getLabel());
    }
  });

  // Second pass: erase only unused labels
  for (auto &lab : labels) {
    StringRef labelName = lab.getKey();
    Block *block = lab.getValue();
    if (!blockAddrLabel.contains(labelName)) {
      // erase the LabelOp inside the block if safe
      if (auto lab = dyn_cast<cir::LabelOp>(&block->front())) {
        lab.erase();
      }
    }
  }

  for (auto goTo : gotos) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(goTo);
    Block *dest = labels[goTo.getLabel()];
    rewriter.create<cir::BrOp>(goTo.getLoc(), dest);
    goTo.erase();
  }
}

void GotoSolverPass::runOnOperation() {
  llvm::TimeTraceScope scope("Goto Solver");
  llvm::SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](cir::FuncOp op) { process(op); });
}

} // namespace

std::unique_ptr<Pass> mlir::createGotoSolverPass() {
  return std::make_unique<GotoSolverPass>();
}
