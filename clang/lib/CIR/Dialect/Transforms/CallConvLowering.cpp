//===- CallConvLowering.cpp - Rewrites functions according to call convs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

FuncOp findFun(mlir::ModuleOp mod, llvm::StringRef name) {
  FuncOp fun;
  mod->walk([&](FuncOp f) {
    if (f.getName() == name) {
      fun = f;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return fun;
}

bool isFuncPointerTy(mlir::Type typ) {
  if (auto ptr = dyn_cast<PointerType>(typ))
    return isa<FuncType>(ptr.getPointee());
  return false;
}

struct CallConvLowering {

  CallConvLowering(LowerModule &mod, mlir::PatternRewriter &rw,
                   mlir::TypeConverter &converter)
    : lowerModule(mod), rewriter(rw), typeConverter(converter) {}

  void lower(Operation *op) {
    rewriter.setInsertionPoint(op);
    if (auto fun = dyn_cast<FuncOp>(op))
      lowerFuncOp(fun);
    else if (auto al = dyn_cast<AllocaOp>(op))
      lowerAllocaOp(al);
    else if (auto glob = dyn_cast<GetGlobalOp>(op))
      lowerGetGlobalOp(glob);
    else if (auto call = dyn_cast<CallOp>(op))
      lowerCallOp(call);
  }

private:

  // don't create new operations once there is no special 
  // conversion for the given type
  mlir::Type convertType(mlir::Type typ) {
    auto newTy = typeConverter.convertType(typ);
    if (newTy != typ)
      return newTy;
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
      
        cir_cconv_assert_or_abort(!::cir::MissingFeatures::ABIFuncPtr(),
                                    "NYI"); 
      }
    }
    lowerModule.rewriteFunctionDefinition(op);
  }

  void lowerAllocaOp(AllocaOp op) {   
    if (auto newEltTy = convertType(op.getAllocaType()))
      rewriter.replaceOpWithNewOp<AllocaOp>(
          op, typeConverter.convertType(op.getResult().getType()), newEltTy,
          op.getName(), op.getAlignmentAttr(), op.getDynAllocSize());
  }

  void lowerGetGlobalOp(GetGlobalOp op) {
    auto resTy = op.getResult().getType();
    if (isFuncPointerTy(resTy))
      if (auto newResTy = convertType(resTy))
        rewriter.replaceOpWithNewOp<GetGlobalOp>(op, newResTy, op.getName());
  }

  void lowerCallOp(CallOp op) {
    auto mod = op->getParentOfType<ModuleOp>();
    if (auto callee = op.getCallee()) {
      if (auto fun = findFun(mod, *callee))
        lowerModule.rewriteFunctionCall(op, fun);
    } else {
      cir_cconv_unreachable("NYI");
    }
  }

private:
  LowerModule &lowerModule;
  mlir::PatternRewriter &rewriter;
  mlir::TypeConverter &typeConverter;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void initTypeConverter(mlir::TypeConverter &converter,
                       mlir::cir::LowerModule &module) {

  converter.addConversion([](mlir::Type typ) -> mlir::Type { return typ; });

  converter.addConversion([&](mlir::cir::FuncType funTy) -> mlir::Type {    
    auto &typs = module.getTypes();  
    return typs.getFunctionType(typs.arrangeFreeFunctionType(funTy));
  });

  converter.addConversion([&](mlir::cir::PointerType ptrTy) -> mlir::Type {
    auto pointee = converter.convertType(ptrTy.getPointee());
    return PointerType::get(module.getMLIRContext(), pointee);
  });

  converter.addConversion([&](mlir::cir::ArrayType arTy) -> mlir::Type {
    auto eltType = converter.convertType(arTy.getEltType());
    return ArrayType::get(module.getMLIRContext(), eltType, arTy.getSize());
  });
}

struct CallConvLoweringPass
    : ::impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  void runOnOperation() override;
  StringRef getArgument() const override { return "cir-call-conv-lowering"; };
};

void CallConvLoweringPass::runOnOperation() {
  auto module = dyn_cast<ModuleOp>(getOperation());
  mlir::PatternRewriter rewriter(module.getContext());
  std::unique_ptr<LowerModule> lowerModule =
      createLowerModule(module, rewriter);

  mlir::TypeConverter converter;
  initTypeConverter(converter, *lowerModule.get());

  CallConvLowering cc(*lowerModule.get(), rewriter, converter);
  module.walk([&](Operation *op) { cc.lower(op); });
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
