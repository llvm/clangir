//===- ABILowering.cpp - Expands ABI-dependent types and operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR ABI lowering pass which expands ABI-dependent
// types and operations to equivalent ABI-independent types and operations.
//
//===----------------------------------------------------------------------===//

#include "TargetLowering/LowerModule.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

#define GEN_PASS_DEF_ABILOWERING
#include "clang/CIR/Dialect/Passes.h.inc"

namespace cir {
namespace {

template <typename Op>
class CIROpABILoweringPattern : public mlir::OpConversionPattern<Op> {
protected:
  mlir::DataLayout *dataLayout;
  cir::LowerModule *lowerModule;

public:
  CIROpABILoweringPattern(mlir::MLIRContext *context,
                          const mlir::TypeConverter &typeConverter,
                          mlir::DataLayout &dataLayout,
                          cir::LowerModule &lowerModule)
      : mlir::OpConversionPattern<Op>(typeConverter, context),
        dataLayout(&dataLayout), lowerModule(&lowerModule) {}
};

#define CIR_ABI_LOWERING_PATTERN(name, operation)                              \
  struct name : CIROpABILoweringPattern<operation> {                           \
    using CIROpABILoweringPattern<operation>::CIROpABILoweringPattern;         \
                                                                               \
    mlir::LogicalResult                                                        \
    matchAndRewrite(operation op, OpAdaptor adaptor,                           \
                    mlir::ConversionPatternRewriter &rewriter) const override; \
  }
CIR_ABI_LOWERING_PATTERN(CIRAllocaOpABILowering, cir::AllocaOp);
CIR_ABI_LOWERING_PATTERN(CIRBaseDataMemberOpABILowering, cir::BaseDataMemberOp);
CIR_ABI_LOWERING_PATTERN(CIRBaseMethodOpABILowering, cir::BaseMethodOp);
CIR_ABI_LOWERING_PATTERN(CIRCastOpABILowering, cir::CastOp);
CIR_ABI_LOWERING_PATTERN(CIRCmpOpABILowering, cir::CmpOp);
CIR_ABI_LOWERING_PATTERN(CIRConstantOpABILowering, cir::ConstantOp);
CIR_ABI_LOWERING_PATTERN(CIRDerivedDataMemberOpABILowering,
                         cir::DerivedDataMemberOp);
CIR_ABI_LOWERING_PATTERN(CIRDerivedMethodOpABILowering, cir::DerivedMethodOp);
CIR_ABI_LOWERING_PATTERN(CIRFuncOpABILowering, cir::FuncOp);
CIR_ABI_LOWERING_PATTERN(CIRGetMethodOpABILowering, cir::GetMethodOp);
CIR_ABI_LOWERING_PATTERN(CIRGetRuntimeMemberOpABILowering,
                         cir::GetRuntimeMemberOp);
CIR_ABI_LOWERING_PATTERN(CIRGlobalOpABILowering, cir::GlobalOp);
#undef CIR_ABI_LOWERING_PATTERN

/// A generic ABI lowering rewrite pattern. This conversion pattern matches any
/// CIR dialect operations with at least one operand or result of an
/// ABI-dependent type. This conversion pattern rewrites the matched operation
/// by replacing all its ABI-dependent operands and results with their
/// lowered counterparts.
class CIRGenericABILoweringPattern : public mlir::ConversionPattern {
public:
  CIRGenericABILoweringPattern(mlir::MLIRContext *context,
                               const mlir::TypeConverter &typeConverter)
      : mlir::ConversionPattern(typeConverter, MatchAnyOpTypeTag(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Do not match on operations that have dedicated ABI lowering rewrite rules
    if (llvm::isa<cir::BaseDataMemberOp, cir::BaseMethodOp, cir::CastOp,
                  cir::CmpOp, cir::ConstantOp, cir::DerivedDataMemberOp,
                  cir::DerivedMethodOp, cir::FuncOp, cir::GetMethodOp,
                  cir::GetRuntimeMemberOp, cir::GlobalOp>(op))
      return mlir::failure();

    const mlir::TypeConverter *typeConverter = getTypeConverter();
    assert(typeConverter &&
           "CIRGenericABILoweringPattern requires a type converter");
    if (typeConverter->isLegal(op)) {
      // The operation does not have any ABI-dependent operands or results, the
      // match fails.
      return mlir::failure();
    }

    assert(op->getNumRegions() == 0 && "CIRGenericABILoweringPattern cannot "
                                       "deal with operations with regions");

    mlir::OperationState loweredOpState(op->getLoc(), op->getName());
    loweredOpState.addOperands(operands);
    loweredOpState.addAttributes(op->getAttrs());
    loweredOpState.addSuccessors(op->getSuccessors());

    // Lower all result types
    llvm::SmallVector<mlir::Type> loweredResultTypes;
    loweredResultTypes.reserve(op->getNumResults());
    for (mlir::Type result : op->getResultTypes())
      loweredResultTypes.push_back(typeConverter->convertType(result));
    loweredOpState.addTypes(loweredResultTypes);

    // Clone the operation with lowered operand types and result types
    mlir::Operation *loweredOp = rewriter.create(loweredOpState);

    rewriter.replaceOp(op, loweredOp);
    return mlir::success();
  }
};

mlir::LogicalResult CIRAllocaOpABILowering::matchAndRewrite(
    cir::AllocaOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type allocaPtrTy = op.getType();
  mlir::Type allocaTy = op.getAllocaType();
  mlir::Type loweredAllocaPtrTy = getTypeConverter()->convertType(allocaPtrTy);
  mlir::Type loweredAllocaTy = getTypeConverter()->convertType(allocaTy);

  cir::AllocaOp loweredOp = rewriter.create<cir::AllocaOp>(
      op.getLoc(), loweredAllocaPtrTy, loweredAllocaTy, op.getName(),
      op.getAlignmentAttr(), /*dynAllocSize=*/adaptor.getDynAllocSize());
  loweredOp.setInit(op.getInit());
  loweredOp.setConstant(op.getConstant());
  loweredOp.setAnnotationsAttr(op.getAnnotationsAttr());

  rewriter.replaceOp(op, loweredOp);
  return mlir::success();
}

mlir::LogicalResult CIRBaseDataMemberOpABILowering::matchAndRewrite(
    cir::BaseDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerModule->getCXXABI().lowerBaseDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRBaseMethodOpABILowering::matchAndRewrite(
    cir::BaseMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult =
      lowerModule->getCXXABI().lowerBaseMethod(op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRCastOpABILowering::matchAndRewrite(
    cir::CastOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type srcTy = op.getSrc().getType();
  assert((mlir::isa<cir::DataMemberType, cir::MethodType>(srcTy)) &&
         "input to bitcast in ABI lowering must be a data member or method");

  switch (op.getKind()) {
  case cir::CastKind::bitcast: {
    mlir::Type destTy = getTypeConverter()->convertType(op.getType());
    mlir::Value loweredResult;
    if (mlir::isa<cir::DataMemberType>(srcTy))
      loweredResult = lowerModule->getCXXABI().lowerDataMemberBitcast(
          op, destTy, adaptor.getSrc(), rewriter);
    else
      loweredResult = lowerModule->getCXXABI().lowerMethodBitcast(
          op, destTy, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, loweredResult);
    return mlir::success();
  }
  case cir::CastKind::member_ptr_to_bool: {
    mlir::Value loweredResult;
    if (mlir::isa<cir::MethodType>(srcTy))
      loweredResult = lowerModule->getCXXABI().lowerMethodToBoolCast(
          op, adaptor.getSrc(), rewriter);
    else
      loweredResult = lowerModule->getCXXABI().lowerDataMemberToBoolCast(
          op, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, loweredResult);
    return mlir::success();
  }
  default:
    break;
  }

  return mlir::failure();
}

mlir::LogicalResult CIRCmpOpABILowering::matchAndRewrite(
    cir::CmpOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto type = op.getLhs().getType();
  assert((mlir::isa<cir::DataMemberType, cir::MethodType>(type)) &&
         "input to cmp in ABI lowering must be a data member or method");

  mlir::Value loweredResult;
  if (mlir::isa<cir::DataMemberType>(type))
    loweredResult = lowerModule->getCXXABI().lowerDataMemberCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);
  else
    loweredResult = lowerModule->getCXXABI().lowerMethodCmp(
        op, adaptor.getLhs(), adaptor.getRhs(), rewriter);

  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRConstantOpABILowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  if (mlir::isa<cir::DataMemberType>(op.getType())) {
    auto dataMember = mlir::cast<cir::DataMemberAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerModule->getCXXABI().lowerDataMemberConstant(
        dataMember, layout, *getTypeConverter());
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }

  if (mlir::isa<cir::MethodType>(op.getType())) {
    auto method = mlir::cast<cir::MethodAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerModule->getCXXABI().lowerMethodConstant(
        method, layout, *getTypeConverter());
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }

  llvm_unreachable("constant operand is not an ABI-dependent type");
}

mlir::LogicalResult CIRDerivedDataMemberOpABILowering::matchAndRewrite(
    cir::DerivedDataMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerModule->getCXXABI().lowerDerivedDataMember(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRDerivedMethodOpABILowering::matchAndRewrite(
    cir::DerivedMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResult = lowerModule->getCXXABI().lowerDerivedMethod(
      op, adaptor.getSrc(), rewriter);
  rewriter.replaceOp(op, loweredResult);
  return mlir::success();
}

mlir::LogicalResult CIRFuncOpABILowering::matchAndRewrite(
    cir::FuncOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  cir::FuncType opFuncType = op.getFunctionType();
  mlir::TypeConverter::SignatureConversion signatureConversion(
      opFuncType.getNumInputs());

  for (const auto &[i, argType] : llvm::enumerate(opFuncType.getInputs())) {
    mlir::Type loweredArgType = getTypeConverter()->convertType(argType);
    if (!loweredArgType)
      return mlir::failure();
    signatureConversion.addInputs(i, loweredArgType);
  }

  mlir::Type loweredResultType =
      getTypeConverter()->convertType(opFuncType.getReturnType());
  if (!loweredResultType)
    return mlir::failure();

  auto loweredFuncType =
      cir::FuncType::get(signatureConversion.getConvertedTypes(),
                         loweredResultType, /*isVarArg=*/opFuncType.isVarArg());

  // Create a new cir.func operation for the ABI-lowered function.
  cir::FuncOp loweredFuncOp = rewriter.cloneWithoutRegions(op);
  loweredFuncOp.setFunctionType(loweredFuncType);
  rewriter.inlineRegionBefore(op.getBody(), loweredFuncOp.getBody(),
                              loweredFuncOp.end());
  if (mlir::failed(rewriter.convertRegionTypes(
          &loweredFuncOp.getBody(), *getTypeConverter(), &signatureConversion)))
    return mlir::failure();

  rewriter.eraseOp(op);
  return mlir::success();
}

mlir::LogicalResult CIRGetMethodOpABILowering::matchAndRewrite(
    cir::GetMethodOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Value loweredResults[2];
  lowerModule->getCXXABI().lowerGetMethod(
      op, loweredResults, adaptor.getMethod(), adaptor.getObject(), rewriter);
  rewriter.replaceOp(op, loweredResults);
  return mlir::success();
}

mlir::LogicalResult CIRGetRuntimeMemberOpABILowering::matchAndRewrite(
    cir::GetRuntimeMemberOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type resTy = getTypeConverter()->convertType(op.getType());
  mlir::Operation *llvmOp = lowerModule->getCXXABI().lowerGetRuntimeMember(
      op, resTy, adaptor.getAddr(), adaptor.getMember(), rewriter);
  rewriter.replaceOp(op, llvmOp);
  return mlir::success();
}

mlir::LogicalResult CIRGlobalOpABILowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type ty = op.getSymType();
  mlir::Type loweredTy = getTypeConverter()->convertType(ty);
  if (!loweredTy)
    return mlir::failure();

  mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());

  mlir::Attribute loweredInit;
  if (mlir::isa<cir::DataMemberType>(ty)) {
    cir::DataMemberAttr init =
        mlir::cast_if_present<cir::DataMemberAttr>(op.getInitialValueAttr());
    loweredInit = lowerModule->getCXXABI().lowerDataMemberConstant(
        init, layout, *getTypeConverter());
  } else if (mlir::isa<cir::MethodType>(ty)) {
    cir::MethodAttr init =
        mlir::cast_if_present<cir::MethodAttr>(op.getInitialValueAttr());
    loweredInit = lowerModule->getCXXABI().lowerMethodConstant(
        init, layout, *getTypeConverter());
  } else {
    llvm_unreachable(
        "inputs to cir.global in ABI lowering must be data member or method");
  }

  auto abiOp = mlir::cast<cir::GlobalOp>(rewriter.clone(*op.getOperation()));
  abiOp.setInitialValueAttr(loweredInit);
  abiOp.setSymType(loweredTy);
  rewriter.replaceOp(op, abiOp);
  return mlir::success();
}

static void prepareABITypeConverter(mlir::TypeConverter &converter,
                                    mlir::DataLayout &dataLayout,
                                    cir::LowerModule &lowerModule) {
  converter.addConversion([&](mlir::Type type) -> mlir::Type { return type; });
  converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    mlir::Type loweredPointeeType = converter.convertType(type.getPointee());
    if (!loweredPointeeType)
      return {};
    return cir::PointerType::get(type.getContext(), loweredPointeeType,
                                 type.getAddrSpace());
  });
  converter.addConversion([&](cir::DataMemberType type) -> mlir::Type {
    mlir::Type abiType = lowerModule.getCXXABI().getDataMemberABIType();
    return converter.convertType(abiType);
  });
  converter.addConversion([&](cir::MethodType type) -> mlir::Type {
    mlir::Type abiType = lowerModule.getCXXABI().getMethodABIType();
    return converter.convertType(abiType);
  });
  converter.addConversion([&](cir::FuncType type) -> mlir::Type {
    llvm::SmallVector<mlir::Type> loweredInputTypes;
    loweredInputTypes.reserve(type.getNumInputs());
    if (mlir::failed(
            converter.convertTypes(type.getInputs(), loweredInputTypes)))
      return {};

    mlir::Type loweredReturnType = converter.convertType(type.getReturnType());
    if (!loweredReturnType)
      return {};

    return cir::FuncType::get(loweredInputTypes, loweredReturnType,
                              /*isVarArg=*/type.getVarArg());
  });
}

static void
populateABIConversionTarget(mlir::ConversionTarget &target,
                            const mlir::TypeConverter &typeConverter) {
  target.addLegalOp<mlir::ModuleOp>();

  // The ABI lowering pass is interested in CIR operations with operands or
  // results of ABI-dependent types, or CIR operations with regions whose block
  // arguments are of ABI-dependent types.
  target.addDynamicallyLegalDialect<cir::CIRDialect>(
      [&typeConverter](mlir::Operation *op) {
        if (!typeConverter.isLegal(op))
          return false;
        return std::all_of(op->getRegions().begin(), op->getRegions().end(),
                           [&typeConverter](mlir::Region &region) {
                             return typeConverter.isLegal(&region);
                           });
      });

  // Some CIR ops needs special checking for legality
  target.addDynamicallyLegalOp<cir::FuncOp>([&typeConverter](cir::FuncOp op) {
    return typeConverter.isLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<cir::GlobalOp>(
      [&typeConverter](cir::GlobalOp op) {
        return typeConverter.isLegal(op.getSymType());
      });
}

//===----------------------------------------------------------------------===//
// The Pass
//===----------------------------------------------------------------------===//

struct ABILoweringPass : ::impl::ABILoweringBase<ABILoweringPass> {
  using ABILoweringBase::ABILoweringBase;

  void runOnOperation() override;
  llvm::StringRef getArgument() const override { return "cir-abi-lowering"; };
};

void ABILoweringPass::runOnOperation() {
  auto module = mlir::cast<mlir::ModuleOp>(getOperation());
  mlir::MLIRContext *ctx = module.getContext();

  // If the triple is not present, e.g. CIR modules parsed from text, we
  // cannot init LowerModule properly.
  assert(!cir::MissingFeatures::makeTripleAlwaysPresent());
  if (!module->hasAttr(cir::CIRDialect::getTripleAttrName())) {
    // If no target triple is available, skip the ABI lowering pass.
    return;
  }

  mlir::PatternRewriter rewriter(ctx);
  std::unique_ptr<cir::LowerModule> lowerModule =
      cir::createLowerModule(module, rewriter);

  mlir::DataLayout dataLayout(module);
  mlir::TypeConverter typeConverter;
  prepareABITypeConverter(typeConverter, dataLayout, *lowerModule);

  mlir::RewritePatternSet patterns(ctx);
  patterns.add<CIRGenericABILoweringPattern>(patterns.getContext(),
                                             typeConverter);
  patterns.add<
      // clang-format off
      CIRAllocaOpABILowering,
      CIRBaseDataMemberOpABILowering,
      CIRBaseMethodOpABILowering,
      CIRCastOpABILowering,
      CIRCmpOpABILowering,
      CIRConstantOpABILowering,
      CIRDerivedDataMemberOpABILowering,
      CIRDerivedMethodOpABILowering,
      CIRFuncOpABILowering,
      CIRGetMethodOpABILowering,
      CIRGetRuntimeMemberOpABILowering,
      CIRGlobalOpABILowering
      // clang-format on
      >(patterns.getContext(), typeConverter, dataLayout, *lowerModule);

  mlir::ConversionTarget target(*ctx);
  populateABIConversionTarget(target, typeConverter);

  if (failed(mlir::applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace
} // namespace cir

std::unique_ptr<mlir::Pass> mlir::createABILoweringPass() {
  return std::make_unique<cir::ABILoweringPass>();
}
