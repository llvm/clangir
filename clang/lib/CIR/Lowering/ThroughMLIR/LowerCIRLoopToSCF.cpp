//====- LowerCIRLoopToSCF.cpp - Lowering from CIR Loop to SCF -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR loop operations to SCF.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cir;
using namespace llvm;

namespace cir {

class SCFLoop {
public:
  SCFLoop(cir::ForOp op, mlir::ConversionPatternRewriter *rewriter)
      : forOp(op), rewriter(rewriter) {}

  int64_t getStep() { return step; }
  mlir::Value getLowerBound() { return lowerBound; }
  mlir::Value getUpperBound() { return upperBound; }
  bool isCanonical() { return canonical; }

  // Returns true if successfully finds both step and induction variable.
  mlir::LogicalResult findStepAndIV();
  cir::CmpOp findCmpOp();
  mlir::Value findIVInitValue();
  void analysis();

  mlir::Value plusConstant(mlir::Value v, mlir::Location loc, int addend);
  void transferToSCFForOp();
  void transformToSCFWhileOp();

private:
  cir::ForOp forOp;
  cir::CmpOp cmpOp;
  mlir::Value ivAddr, lowerBound = nullptr, upperBound = nullptr;
  mlir::ConversionPatternRewriter *rewriter;
  int64_t step = 0;
  bool canonical = true;
};

class SCFWhileLoop {
public:
  SCFWhileLoop(cir::WhileOp op, cir::WhileOp::Adaptor adaptor,
               mlir::ConversionPatternRewriter *rewriter)
      : whileOp(op), adaptor(adaptor), rewriter(rewriter) {}
  mlir::scf::WhileOp transferToSCFWhileOp();

private:
  cir::WhileOp whileOp;
  cir::WhileOp::Adaptor adaptor;
  mlir::scf::WhileOp scfWhileOp;
  mlir::ConversionPatternRewriter *rewriter;
};

class SCFDoLoop {
public:
  SCFDoLoop(cir::DoWhileOp op, cir::DoWhileOp::Adaptor adaptor,
            mlir::ConversionPatternRewriter *rewriter)
      : DoOp(op), adaptor(adaptor), rewriter(rewriter) {}
  void transferToSCFWhileOp();

private:
  cir::DoWhileOp DoOp;
  cir::DoWhileOp::Adaptor adaptor;
  mlir::ConversionPatternRewriter *rewriter;
};

static int64_t getConstant(cir::ConstantOp op) {
  auto attr = op.getValue();
  const auto intAttr = mlir::cast<cir::IntAttr>(attr);
  return intAttr.getValue().getSExtValue();
}

mlir::LogicalResult SCFLoop::findStepAndIV() {
  auto *stepBlock =
      (forOp.maybeGetStep() ? &forOp.maybeGetStep()->front() : nullptr);
  assert(stepBlock && "Can not find step block");

  // Try to match "iv = load addr; ++iv; store iv, addr; yield" to find step.
  // We should match the exact pattern, in case there's something unexpected:
  // we must rule out cases like `for (int i = 0; i < n; i++, printf("\n"))`.
  auto &oplist = stepBlock->getOperations();

  auto iterator = oplist.begin();

  // We might find constants at beginning. Skip them.
  // We could have hoisted them outside the for loop in previous passes, but
  // it hasn't been done yet.
  while (iterator != oplist.end() && isa<ConstantOp>(*iterator))
    ++iterator;

  if (iterator == oplist.end())
    return mlir::failure();

  auto load = dyn_cast<LoadOp>(*iterator);
  if (!load)
    return mlir::failure();

  // We assume this is the address of induction variable (IV). The operations
  // that come next will check if that's true.
  mlir::Value addr = load.getAddr();
  mlir::Value iv = load.getResult();

  // Then we try to match either "++IV" or "IV += n". Same for reversed loops.
  if (++iterator == oplist.end())
    return mlir::failure();

  mlir::Operation &arith = *iterator;

  if (auto unary = dyn_cast<UnaryOp>(arith)) {
    // Not operating on induction variable. Fail.
    if (unary.getInput() != iv)
      return mlir::failure();

    if (unary.getKind() == UnaryOpKind::Inc)
      step = 1;
    else if (unary.getKind() == UnaryOpKind::Dec)
      step = -1;
    else
      return mlir::failure();
  }

  if (auto binary = dyn_cast<BinOp>(arith)) {
    if (binary.getLhs() != iv)
      return mlir::failure();

    mlir::Value value = binary.getRhs();
    if (auto constValue = dyn_cast<ConstantOp>(value.getDefiningOp());
        isa<IntAttr>(constValue.getValue()))
      step = getConstant(constValue);

    if (binary.getKind() == BinOpKind::Add)
      ; // Nothing to do. Step has been calculated above.
    else if (binary.getKind() == BinOpKind::Sub)
      step = -step;
    else
      return mlir::failure();
  }

  // Check whether we immediately store this value into the appropriate place.
  if (++iterator == oplist.end())
    return mlir::failure();

  auto store = dyn_cast<StoreOp>(*iterator);
  if (!store || store.getAddr() != addr ||
      store.getValue() != arith.getResult(0))
    return mlir::failure();

  if (++iterator == oplist.end())
    return mlir::failure();

  // Finally, this should precede a yield with nothing in between.
  bool success = isa<YieldOp>(*iterator);

  // Remember to update analysis information.
  if (success)
    ivAddr = addr;

  return success ? mlir::success() : mlir::failure();
}

static bool isIVLoad(mlir::Operation *op, mlir::Value IVAddr) {
  if (!op)
    return false;
  if (isa<cir::LoadOp>(op)) {
    if (!op->getOperand(0))
      return false;
    if (op->getOperand(0) == IVAddr)
      return true;
  }
  return false;
}

cir::CmpOp SCFLoop::findCmpOp() {
  cmpOp = nullptr;
  for (auto *user : ivAddr.getUsers()) {
    if (user->getParentRegion() != &forOp.getCond())
      continue;
    if (auto loadOp = dyn_cast<cir::LoadOp>(*user)) {
      if (!loadOp->hasOneUse())
        continue;
      if (auto op = dyn_cast<cir::CmpOp>(*loadOp->user_begin())) {
        cmpOp = op;
        break;
      }
    }
  }
  if (!cmpOp)
    return nullptr;

  auto type = cmpOp.getLhs().getType();
  if (!mlir::isa<cir::IntType>(type))
    return nullptr;

  auto *lhsDefOp = cmpOp.getLhs().getDefiningOp();
  if (!lhsDefOp)
    return nullptr;
  if (!isIVLoad(lhsDefOp, ivAddr))
    return nullptr;

  if (cmpOp.getKind() != cir::CmpOpKind::le &&
      cmpOp.getKind() != cir::CmpOpKind::lt)
    return nullptr;

  return cmpOp;
}

mlir::Value SCFLoop::plusConstant(mlir::Value V, mlir::Location loc,
                                  int addend) {
  auto type = V.getType();
  auto c1 = rewriter->create<mlir::arith::ConstantOp>(
      loc, mlir::IntegerAttr::get(type, addend));
  return rewriter->create<mlir::arith::AddIOp>(loc, V, c1);
}

// Return IV initial value by searching the store before the loop.
// The operations before the loop have been transferred to MLIR.
// So we need to go through getRemappedValue to find the value.
mlir::Value SCFLoop::findIVInitValue() {
  auto remapAddr = rewriter->getRemappedValue(ivAddr);
  if (!remapAddr)
    return nullptr;
  if (!remapAddr.hasOneUse())
    return nullptr;
  auto memrefStore = dyn_cast<mlir::memref::StoreOp>(*remapAddr.user_begin());
  if (!memrefStore)
    return nullptr;
  return memrefStore->getOperand(0);
}

void SCFLoop::analysis() {
  canonical = mlir::succeeded(findStepAndIV());
  if (!canonical)
    return;

  cmpOp = findCmpOp();
  if (!cmpOp) {
    canonical = false;
    return;
  }

  auto ivInit = findIVInitValue();
  if (!ivInit) {
    canonical = false;
    return;
  }

  // The loop end value should be hoisted out of loop by -cir-mlir-scf-prepare.
  // So we could get the value by getRemappedValue.
  auto ivEndBound = rewriter->getRemappedValue(cmpOp.getRhs());
  // If the loop end bound is not loop invariant and can't be hoisted,
  // then this is not a canonical loop.
  if (!ivEndBound) {
    canonical = false;
    return;
  }

  if (step > 0) {
    lowerBound = ivInit;
    if (cmpOp.getKind() == cir::CmpOpKind::lt)
      upperBound = ivEndBound;
    else if (cmpOp.getKind() == cir::CmpOpKind::le)
      upperBound = plusConstant(ivEndBound, cmpOp.getLoc(), 1);
  }
  if (!lowerBound || !upperBound)
    canonical = false;
}

void SCFLoop::transferToSCFForOp() {
  auto ub = getUpperBound();
  auto lb = getLowerBound();
  auto loc = forOp.getLoc();
  auto type = lb.getType();
  auto step = rewriter->create<mlir::arith::ConstantOp>(
      loc, mlir::IntegerAttr::get(type, getStep()));
  auto scfForOp = rewriter->create<mlir::scf::ForOp>(loc, lb, ub, step);
  SmallVector<mlir::Value> bbArg;
  rewriter->eraseOp(&scfForOp.getBody()->back());
  rewriter->inlineBlockBefore(&forOp.getBody().front(), scfForOp.getBody(),
                              scfForOp.getBody()->end(), bbArg);
  scfForOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<cir::BreakOp>(op) || isa<cir::ContinueOp>(op) || isa<cir::IfOp>(op))
      llvm_unreachable(
          "Not support lowering loop with break, continue or if yet");
    // Replace the IV usage to scf loop induction variable.
    if (isIVLoad(op, ivAddr)) {
      // Replace CIR IV load with arith.addi scf.IV, 0.
      // The replacement makes the SCF IV can be automatically propogated
      // by OpAdaptor for individual IV user lowering.
      // The redundant arith.addi can be removed by later MLIR passes.
      rewriter->setInsertionPoint(op);
      auto newIV = plusConstant(scfForOp.getInductionVar(), loc, 0);
      rewriter->replaceOp(op, newIV.getDefiningOp());
    }
    return mlir::WalkResult::advance();
  });
}

void SCFLoop::transformToSCFWhileOp() {
  auto scfWhileOp = rewriter->create<mlir::scf::WhileOp>(
      forOp->getLoc(), forOp->getResultTypes(), mlir::ValueRange());
  rewriter->createBlock(&scfWhileOp.getBefore());
  rewriter->createBlock(&scfWhileOp.getAfter());

  rewriter->inlineBlockBefore(&forOp.getCond().front(),
                              scfWhileOp.getBeforeBody(),
                              scfWhileOp.getBeforeBody()->end());
  rewriter->inlineBlockBefore(&forOp.getBody().front(),
                              scfWhileOp.getAfterBody(),
                              scfWhileOp.getAfterBody()->end());
  // There will be a yield after the `for` body.
  // We should delete it.
  auto yield = mlir::cast<YieldOp>(scfWhileOp.getAfterBody()->back());
  rewriter->eraseOp(yield);

  rewriter->inlineBlockBefore(&forOp.getStep().front(),
                              scfWhileOp.getAfterBody(),
                              scfWhileOp.getAfterBody()->end());
}

mlir::scf::WhileOp SCFWhileLoop::transferToSCFWhileOp() {
  auto scfWhileOp = rewriter->create<mlir::scf::WhileOp>(
      whileOp->getLoc(), whileOp->getResultTypes(), adaptor.getOperands());
  rewriter->createBlock(&scfWhileOp.getBefore());
  rewriter->createBlock(&scfWhileOp.getAfter());
  rewriter->inlineBlockBefore(&whileOp.getCond().front(),
                              scfWhileOp.getBeforeBody(),
                              scfWhileOp.getBeforeBody()->end());
  rewriter->inlineBlockBefore(&whileOp.getBody().front(),
                              scfWhileOp.getAfterBody(),
                              scfWhileOp.getAfterBody()->end());
  return scfWhileOp;
}

void SCFDoLoop::transferToSCFWhileOp() {

  auto beforeBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange args) {
    auto *newBlock = builder.getBlock();
    rewriter->mergeBlocks(&DoOp.getBody().front(), newBlock);
    auto *yieldOp = newBlock->getTerminator();
    rewriter->mergeBlocks(&DoOp.getCond().front(), newBlock,
                          yieldOp->getResults());
    rewriter->eraseOp(yieldOp);
  };
  auto afterBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::ValueRange args) {
    rewriter->create<mlir::scf::YieldOp>(loc, args);
  };

  rewriter->create<mlir::scf::WhileOp>(DoOp.getLoc(), DoOp->getResultTypes(),
                                       adaptor.getOperands(), beforeBuilder,
                                       afterBuilder);
}

class CIRForOpLowering : public mlir::OpConversionPattern<cir::ForOp> {
public:
  using OpConversionPattern<cir::ForOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ForOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SCFLoop loop(op, &rewriter);
    loop.analysis();
    if (!loop.isCanonical()) {
      loop.transformToSCFWhileOp();
      rewriter.eraseOp(op);
      return mlir::success();
    }

    loop.transferToSCFForOp();
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRWhileOpLowering : public mlir::OpConversionPattern<cir::WhileOp> {
  void rewriteContinue(mlir::scf::WhileOp whileOp,
                       mlir::ConversionPatternRewriter &rewriter) const {
    // Collect all ContinueOp inside this while.
    llvm::SmallVector<cir::ContinueOp> continues;
    whileOp->walk([&](mlir::Operation *op) {
      if (auto continueOp = dyn_cast<ContinueOp>(op))
        continues.push_back(continueOp);
    });

    if (continues.empty())
      return;

    for (auto continueOp : continues) {
      // When the break is under an IfOp, a direct replacement of `scf.yield`
      // won't work: the yield would jump out of that IfOp instead. We might
      // need to change the whileOp itself to achieve the same effect.
      for (mlir::Operation *parent = continueOp->getParentOp();
           parent != whileOp; parent = parent->getParentOp()) {
        if (isa<mlir::scf::IfOp>(parent) || isa<cir::IfOp>(parent))
          llvm_unreachable("NYI");
      }

      // Operations after this break has to be removed.
      for (mlir::Operation *runner = continueOp->getNextNode(); runner;) {
        mlir::Operation *next = runner->getNextNode();
        runner->erase();
        runner = next;
      }

      // Blocks after this break also has to be removed.
      for (mlir::Block *block = continueOp->getBlock()->getNextNode(); block;) {
        mlir::Block *next = block->getNextNode();
        block->erase();
        block = next;
      }
    }
  }

public:
  using OpConversionPattern<cir::WhileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::WhileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SCFWhileLoop loop(op, adaptor, &rewriter);
    auto whileOp = loop.transferToSCFWhileOp();
    rewriteContinue(whileOp, rewriter);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRDoOpLowering : public mlir::OpConversionPattern<cir::DoWhileOp> {
public:
  using OpConversionPattern<cir::DoWhileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::DoWhileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SCFDoLoop loop(op, adaptor, &rewriter);
    loop.transferToSCFWhileOp();
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRConditionOpLowering
    : public mlir::OpConversionPattern<cir::ConditionOp> {
public:
  using OpConversionPattern<cir::ConditionOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::ConditionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::WhileOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
              op, adaptor.getCondition(), parentOp->getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

void populateCIRLoopToSCFConversionPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::TypeConverter &converter) {
  patterns.add<CIRForOpLowering, CIRWhileOpLowering, CIRConditionOpLowering,
               CIRDoOpLowering>(converter, patterns.getContext());
}

} // namespace cir
