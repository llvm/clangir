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
  };
CIR_ABI_LOWERING_PATTERN(CIRCastOpABILowering, cir::CastOp)
CIR_ABI_LOWERING_PATTERN(CIRGlobalOpABILowering, cir::GlobalOp)
CIR_ABI_LOWERING_PATTERN(CIRConstantOpABILowering, cir::ConstantOp)
CIR_ABI_LOWERING_PATTERN(CIRBaseDataMemberOpABILowering, cir::BaseDataMemberOp)
CIR_ABI_LOWERING_PATTERN(CIRBaseMethodOpABILowering, cir::BaseMethodOp)
CIR_ABI_LOWERING_PATTERN(CIRCmpOpABILowering, cir::CmpOp)
CIR_ABI_LOWERING_PATTERN(CIRDerivedDataMemberOpABILowering,
                         cir::DerivedDataMemberOp)
CIR_ABI_LOWERING_PATTERN(CIRDerivedMethodOpABILowering, cir::DerivedMethodOp)
CIR_ABI_LOWERING_PATTERN(CIRGetMethodOpABILowering, cir::GetMethodOp)
CIR_ABI_LOWERING_PATTERN(CIRGetRuntimeMemberOpABILowering,
                         cir::GetRuntimeMemberOp)
#undef CIR_ABI_LOWERING_PATTERN

mlir::LogicalResult CIRCastOpABILowering::matchAndRewrite(
    cir::CastOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  switch (op.getKind()) {
  case cir::CastKind::bitcast: {
    if (!mlir::isa<cir::DataMemberType, cir::MethodType>(op.getSrc().getType()))
      break;

    mlir::Type destTy = getTypeConverter()->convertType(op.getType());
    mlir::Value loweredResult;
    if (mlir::isa<cir::DataMemberType>(op.getSrc().getType()))
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
    if (mlir::isa<cir::MethodType>(op.getSrc().getType()))
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

mlir::LogicalResult CIRGlobalOpABILowering::matchAndRewrite(
    cir::GlobalOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  std::optional<mlir::Attribute> init = op.getInitialValue();
  if (!init.has_value())
    return mlir::failure();

  if (auto dataMemberAttr = mlir::dyn_cast<cir::DataMemberAttr>(*init)) {
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerModule->getCXXABI().lowerDataMemberConstant(
        dataMemberAttr, layout, *typeConverter);
    auto abiOp = mlir::cast<GlobalOp>(rewriter.clone(*op.getOperation()));
    abiOp.setInitialValueAttr(abiValue);
    abiOp.setSymType(abiValue.getType());
    rewriter.replaceOp(op, abiOp);
    return mlir::success();
  }

  return mlir::success();
}

mlir::LogicalResult CIRConstantOpABILowering::matchAndRewrite(
    cir::ConstantOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (mlir::isa<cir::DataMemberType>(op.getType())) {
    auto dataMember = mlir::cast<cir::DataMemberAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerModule->getCXXABI().lowerDataMemberConstant(
        dataMember, layout, *typeConverter);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }
  if (mlir::isa<cir::MethodType>(op.getType())) {
    auto method = mlir::cast<cir::MethodAttr>(op.getValue());
    mlir::DataLayout layout(op->getParentOfType<mlir::ModuleOp>());
    mlir::TypedAttr abiValue = lowerModule->getCXXABI().lowerMethodConstant(
        method, layout, *typeConverter);
    rewriter.replaceOpWithNewOp<ConstantOp>(op, abiValue);
    return mlir::success();
  }

  return mlir::failure();
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

mlir::LogicalResult CIRCmpOpABILowering::matchAndRewrite(
    cir::CmpOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto type = op.getLhs().getType();
  if (!mlir::isa<cir::DataMemberType, cir::MethodType>(type))
    return mlir::failure();

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

static void prepareABITypeConverter(mlir::TypeConverter &converter,
                                    mlir::DataLayout &dataLayout,
                                    cir::LowerModule &lowerModule) {
  converter.addConversion([&](mlir::Type type) -> mlir::Type { return type; });
  converter.addConversion([&](cir::DataMemberType type) -> mlir::Type {
    mlir::Type abiType =
        lowerModule.getCXXABI().lowerDataMemberType(type, converter);
    return converter.convertType(abiType);
  });
  converter.addConversion([&](cir::MethodType type) -> mlir::Type {
    mlir::Type abiType =
        lowerModule.getCXXABI().lowerMethodType(type, converter);
    return converter.convertType(abiType);
  });
}

static void populateABILoweringPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::TypeConverter &converter,
                                        mlir::DataLayout &dataLayout,
                                        cir::LowerModule &lowerModule) {
  patterns.add<
      // clang-format off
    CIRBaseDataMemberOpABILowering,
    CIRBaseMethodOpABILowering,
    CIRCastOpABILowering,
    CIRCmpOpABILowering,
    CIRConstantOpABILowering,
    CIRDerivedDataMemberOpABILowering,
    CIRDerivedMethodOpABILowering,
    CIRGetMethodOpABILowering,
    CIRGetRuntimeMemberOpABILowering,
    CIRGlobalOpABILowering
      // clang-format on
      >(patterns.getContext(), converter, dataLayout, lowerModule);
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

  mlir::PatternRewriter rewriter(ctx);
  std::unique_ptr<cir::LowerModule> lowerModule =
      cir::createLowerModule(module, rewriter);

  mlir::DataLayout dataLayout(module);
  mlir::TypeConverter converter;
  prepareABITypeConverter(converter, dataLayout, *lowerModule);

  mlir::RewritePatternSet patterns(ctx);
  populateABILoweringPatterns(patterns, converter, dataLayout, *lowerModule);

  mlir::ConversionTarget target(*ctx);
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<cir::CIRDialect>();

  // TODO: mark operations working on member pointers as illegal

  // Illegal: base-to-derived and derived-to-base conversions on member pointers
  target.addIllegalOp<cir::BaseDataMemberOp, cir::BaseMethodOp,
                      cir::DerivedDataMemberOp, cir::DerivedMethodOp>();
  // Illegal: indirection on member pointers
  target.addIllegalOp<cir::GetRuntimeMemberOp, cir::GetMethodOp>();

  if (failed(mlir::applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace
} // namespace cir

std::unique_ptr<mlir::Pass> mlir::createABILoweringPass() {
  return std::make_unique<cir::ABILoweringPass>();
}
