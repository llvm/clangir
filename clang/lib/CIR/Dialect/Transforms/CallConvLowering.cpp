//===- CallConvLowering.cpp - Rewrites functions according to call convs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <map>
#include "TargetLowering/LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

#include "llvm/Support/TimeProfiler.h"

namespace mlir {
namespace cir {

FuncType getFuncPointerTy(mlir::Type typ) {
  if (auto ptr = dyn_cast<PointerType>(typ))
    return dyn_cast<FuncType>(ptr.getPointee());
  return {};
}

bool isFuncPointerTy(mlir::Type typ) {
  return (bool)getFuncPointerTy(typ);  
}

struct CallConvLowering {

  CallConvLowering(ModuleOp module) 
    : rewriter(module.getContext())
    , lowerModule(createLowerModule(module, rewriter)) {}
  
  void lower(Operation *op) {
   if (auto glob = dyn_cast<GetGlobalOp>(op))
      rewriteGetGlobalOp(glob);
    else if (auto fun = dyn_cast<FuncOp>(op))
      lowerFuncOp(fun);    
    else if (auto call = dyn_cast<CallOp>(op))
      lowerCallOp(call);
  }

private:

  FuncOp getFuncOp(mlir::ModuleOp mod, llvm::StringRef name) {
    auto it = funs.find(name);
    if (it != funs.end())
      return it->second;

    FuncOp fun;
    mod->walk([&](FuncOp f) {
      if (f.getName() == name) {
        fun = f;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    funs[name] = fun;
    return fun;
  }

  FuncType convert(FuncType t) {
    auto &typs = lowerModule->getTypes();
    return typs.getFunctionType(typs.arrangeFreeFunctionType(t));
  }

  mlir::Type convert(mlir::Type t) {
    if (auto fTy = getFuncPointerTy(t)) 
      return PointerType::get(rewriter.getContext(), convert(fTy));
    return t;
  }

  CastOp bitcast(Value src, Type newTy) {
    if (src.getType() != newTy) {
      auto cast = rewriter.create<CastOp>(src.getLoc(), newTy, CastKind::bitcast, src);
      rewriter.replaceAllUsesExcept(src, cast, {cast});
    }
    return {};
  }

  void lowerFuncOp(FuncOp op) {
    // Fail the pass on unimplemented function users
    const auto module = op->getParentOfType<mlir::ModuleOp>();
    auto calls = op.getSymbolUses(module);
    if (calls.has_value()) {
      for (auto call : calls.value()) {
        if (isa<GetGlobalOp, CallOp>(call.getUser()))
          continue;

        cir_cconv_assert_or_abort(!::cir::MissingFeatures::ABIFuncPtr(), "NYI");
      }
    }
    lowerModule->rewriteFunctionDefinition(op);
  }

  void rewriteGetGlobalOp(GetGlobalOp op) {
    auto resTy = op.getResult().getType();
    if (isFuncPointerTy(resTy)) {
      rewriter.setInsertionPoint(op);
      auto newOp = rewriter.replaceOpWithNewOp<GetGlobalOp>(op, convert(resTy), op.getName());     
      rewriter.setInsertionPointAfter(newOp);
      bitcast(newOp, resTy); 
    }
  }

  void lowerCallOp(CallOp op) {
    auto mod = op->getParentOfType<ModuleOp>();
    if (auto callee = op.getCallee()) {
      if (auto fun = getFuncOp(mod, *callee))
        lowerModule->rewriteFunctionCall(op, fun);
    } else if (op.isIndirect()) {
      rewriter.setInsertionPoint(op);
      auto typ = op.getIndirectCall().getType();
      if (isFuncPointerTy(typ)) {        
        bitcast(op.getIndirectCall(), convert(typ));
        cir_cconv_unreachable("Indirect calls NYI");
      }
    } else {
      cir_cconv_unreachable("NYI");
    }
  }

private:
  mlir::PatternRewriter rewriter;
  std::unique_ptr<LowerModule> lowerModule;
  std::map<StringRef, FuncOp> funs;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct CallConvLoweringPass
    : ::impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  void runOnOperation() override;
  StringRef getArgument() const override { return "cir-call-conv-lowering"; };
};

void CallConvLoweringPass::runOnOperation() {
  auto module = dyn_cast<ModuleOp>(getOperation());
  CallConvLowering cc(module);
  module.walk([&](Operation *op) { cc.lower(op); });  
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
