//===- CallConvLowering.cpp - Rewrites functions according to call convs --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetLowering/LowerModule.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

#include "llvm/Support/TimeProfiler.h"

namespace mlir {
namespace cir {

FuncType lowerFuncType(LowerModule& mod, FuncType ftyp) {
  auto& typs = mod.getTypes();
  auto& info = typs.arrangeFreeFunctionType(ftyp);
  return typs.getFunctionType(info);
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

class CCFuncOpLowering : public mlir::OpRewritePattern<FuncOp> { 
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LowerModule& lowerModule;

public:
  CCFuncOpLowering(LowerModule& mod, mlir::MLIRContext *context)      
      : OpRewritePattern(context)
      , lowerModule(mod) {}

  LogicalResult matchAndRewrite(FuncOp op,  
                                PatternRewriter &rewriter) const final {    
    llvm::TimeTraceScope scope("Call Conv Lowering Pass", op.getSymName().str());
    const auto module = op->getParentOfType<mlir::ModuleOp>();

    // Rewrite function calls before definitions. This should be done before
    // lowering the definition.
    auto calls = op.getSymbolUses(module);
    if (calls.has_value()) {
      for (auto call : calls.value()) {
        // // FIXME(cir): Function pointers are ignored.
        // if (isa<GetGlobalOp>(call.getUser())) {
        //   cir_cconv_assert_or_abort(!::cir::MissingFeatures::ABIFuncPtr(),
        //                             "NYI");
        //   continue;
        // }

        auto callOp = dyn_cast_or_null<CallOp>(call.getUser());
        // if (!callOp)
        //   cir_cconv_unreachable("NYI empty callOp");
        if (auto callOp = dyn_cast_or_null<CallOp>(call.getUser()))
          if (lowerModule.rewriteFunctionCall(callOp, op).failed())
            return failure();
      }
    }

    // TODO(cir): Instead of re-emmiting every load and store, bitcast arguments
    // and return values to their ABI-specific counterparts when possible.
    return lowerModule.rewriteFunctionDefinition(op);      
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

void initTypeConverter(mlir::TypeConverter& converter,
                       mlir::cir::LowerModule& module) {
   
  converter.addConversion([](mlir::Type typ) -> mlir::Type { return typ; });

  converter.addConversion([&](mlir::cir::FuncType funTy) -> mlir::Type {
    return lowerFuncType(module, funTy);
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

class CCGetGlobalOpLowering 
    : public mlir::OpConversionPattern<mlir::cir::GetGlobalOp> {

public:
  CCGetGlobalOpLowering(const mlir::TypeConverter &typeConverter,
                        mlir::MLIRContext *context)
      : OpConversionPattern<mlir::cir::GetGlobalOp>(typeConverter, context) 
      {}
  
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = op.getResult().getType();
    if (auto ptrTy = dyn_cast<PointerType>(resTy)) {
      if (isa<FuncType>(ptrTy.getPointee())) {
        rewriter.replaceOpWithNewOp<GetGlobalOp>(op,
            getTypeConverter()->convertType(resTy),
            op.getName());

        return success();
      }
    }
  
    return failure();
  }

};

class CCAllocaOpLowering : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {

public:
  CCAllocaOpLowering(const mlir::TypeConverter &typeConverter,
                     mlir::MLIRContext *context)
      : OpConversionPattern<mlir::cir::AllocaOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto eltTy = getTypeConverter()->convertType(op.getAllocaType());
    if (op.getAllocaType() != eltTy) {
      rewriter.replaceOpWithNewOp<AllocaOp>(op,
          getTypeConverter()->convertType(op.getResult().getType()),
          eltTy,
          op.getName(),
          op.getAlignmentAttr(),
          op.getDynAllocSize());
      return success();
    }

    return failure();
  }
};

struct CallConvLoweringPass
    : ::impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  void runOnOperation() override;
  StringRef getArgument() const override { return "cir-call-conv-lowering"; };
};

void populateCallConvLoweringPassPatterns(const mlir::TypeConverter& converter,
                                          LowerModule& mod, 
                                          RewritePatternSet &patterns) {  
  patterns.add<CCFuncOpLowering>(mod, patterns.getContext());  
  patterns.add<CCGetGlobalOpLowering, CCAllocaOpLowering>(converter, patterns.getContext());   
}

void CallConvLoweringPass::runOnOperation() {
  auto module = dyn_cast<ModuleOp>(getOperation());
  mlir::PatternRewriter rewriter(module.getContext());
  std::unique_ptr<LowerModule> lowerModule = 
      createLowerModule(module, rewriter);
  
  mlir::TypeConverter converter;
  initTypeConverter(converter, *lowerModule.get());

  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  populateCallConvLoweringPassPatterns(converter, *lowerModule.get(), patterns);

  // Collect operations to be considered by the pass.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](Operation*op) {
    if (isa<AllocaOp, FuncOp, GetGlobalOp>(op))
      ops.push_back(op);
  });

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  // Apply patterns.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace cir

std::unique_ptr<Pass> createCallConvLoweringPass() {
  return std::make_unique<cir::CallConvLoweringPass>();
}

} // namespace mlir
