//====- LowerCIRToMLIR.cpp - Lowering from CIR to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to MLIR.
//
//===----------------------------------------------------------------------===//

#include "LowerToMLIRHelpers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cir;
using namespace llvm;

namespace cir {

class CIRReturnLowering
    : public mlir::OpConversionPattern<mlir::cir::ReturnOp> {
public:
  using OpConversionPattern<mlir::cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::math::MathDialect,
                    mlir::vector::VectorDialect>();
  }
  void runOnOperation() final;

  virtual StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRCallOpLowering : public mlir::OpConversionPattern<mlir::cir::CallOp> {
public:
  using OpConversionPattern<mlir::cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, mlir::SymbolRefAttr::get(op), types, adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRAllocaOpLowering
    : public mlir::OpConversionPattern<mlir::cir::AllocaOp> {
public:
  using OpConversionPattern<mlir::cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = adaptor.getAllocaType();
    auto mlirType = getTypeConverter()->convertType(type);

    // FIXME: Some types can not be converted yet (e.g. struct)
    if (!mlirType)
      return mlir::LogicalResult::failure();

    auto memreftype = mlir::MemRefType::get({}, mlirType);
    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memreftype,
                                                        op.getAlignmentAttr());
    return mlir::LogicalResult::success();
  }
};

// Find base and indices from memref.reinterpret_cast
// and put it into eraseList.
static bool findBaseAndIndices(mlir::Value addr, mlir::Value &base,
                               SmallVector<mlir::Value> &indices,
                               SmallVector<mlir::Operation *> &eraseList,
                               mlir::ConversionPatternRewriter &rewriter) {
  while (mlir::Operation *addrOp = addr.getDefiningOp()) {
    if (!isa<mlir::memref::ReinterpretCastOp>(addrOp))
      break;
    indices.push_back(addrOp->getOperand(1));
    addr = addrOp->getOperand(0);
    eraseList.push_back(addrOp);
  }
  base = addr;
  if (indices.size() == 0)
    return false;
  std::reverse(indices.begin(), indices.end());
  return true;
}

// For memref.reinterpret_cast has multiple users, erasing the operation
// after the last load or store been generated.
static void eraseIfSafe(mlir::Value oldAddr, mlir::Value newAddr,
                        SmallVector<mlir::Operation *> &eraseList,
                        mlir::ConversionPatternRewriter &rewriter) {
  unsigned oldUsedNum =
      std::distance(oldAddr.getUses().begin(), oldAddr.getUses().end());
  unsigned newUsedNum = 0;
  for (auto *user : newAddr.getUsers()) {
    if (isa<mlir::memref::LoadOp>(*user) || isa<mlir::memref::StoreOp>(*user))
      ++newUsedNum;
  }
  if (oldUsedNum == newUsedNum) {
    for (auto op : eraseList)
      rewriter.eraseOp(op);
  }
}

class CIRLoadOpLowering : public mlir::OpConversionPattern<mlir::cir::LoadOp> {
public:
  using OpConversionPattern<mlir::cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value base;
    SmallVector<mlir::Value> indices;
    SmallVector<mlir::Operation *> eraseList;
    if (findBaseAndIndices(adaptor.getAddr(), base, indices, eraseList,
                           rewriter)) {
      rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, base, indices);
      eraseIfSafe(op.getAddr(), base, eraseList, rewriter);
    } else
      rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRStoreOpLowering
    : public mlir::OpConversionPattern<mlir::cir::StoreOp> {
public:
  using OpConversionPattern<mlir::cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value base;
    SmallVector<mlir::Value> indices;
    SmallVector<mlir::Operation *> eraseList;
    if (findBaseAndIndices(adaptor.getAddr(), base, indices, eraseList,
                           rewriter)) {
      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, adaptor.getValue(),
                                                         base, indices);
      eraseIfSafe(op.getAddr(), base, eraseList, rewriter);
    } else
      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, adaptor.getValue(),
                                                         adaptor.getAddr());
    return mlir::LogicalResult::success();
  }
};

class CIRCosOpLowering : public mlir::OpConversionPattern<mlir::cir::CosOp> {
public:
  using OpConversionPattern<mlir::cir::CosOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CosOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::CosOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRSqrtOpLowering : public mlir::OpConversionPattern<mlir::cir::SqrtOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::SqrtOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SqrtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::SqrtOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRFAbsOpLowering : public mlir::OpConversionPattern<mlir::cir::FAbsOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::FAbsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FAbsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::AbsFOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRFloorOpLowering
    : public mlir::OpConversionPattern<mlir::cir::FloorOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::FloorOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FloorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::FloorOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRCeilOpLowering : public mlir::OpConversionPattern<mlir::cir::CeilOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::CeilOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CeilOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::CeilOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRLog10OpLowering
    : public mlir::OpConversionPattern<mlir::cir::Log10Op> {
public:
  using mlir::OpConversionPattern<mlir::cir::Log10Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::Log10Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::Log10Op>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRLogOpLowering : public mlir::OpConversionPattern<mlir::cir::LogOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::LogOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::LogOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::LogOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRLog2OpLowering : public mlir::OpConversionPattern<mlir::cir::Log2Op> {
public:
  using mlir::OpConversionPattern<mlir::cir::Log2Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::Log2Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::Log2Op>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRRoundOpLowering
    : public mlir::OpConversionPattern<mlir::cir::RoundOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::RoundOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::RoundOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::RoundOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRExpOpLowering : public mlir::OpConversionPattern<mlir::cir::ExpOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::ExpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ExpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::ExpOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRShiftOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ShiftOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::ShiftOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ShiftOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cirAmtTy =
        mlir::dyn_cast<mlir::cir::IntType>(op.getAmount().getType());
    auto cirValTy = mlir::dyn_cast<mlir::cir::IntType>(op.getValue().getType());
    auto mlirTy = getTypeConverter()->convertType(op.getType());
    mlir::Value amt = adaptor.getAmount();
    mlir::Value val = adaptor.getValue();

    assert(cirValTy && cirAmtTy && "non-integer shift is NYI");
    assert(cirValTy == op.getType() && "inconsistent operands' types NYI");

    // Ensure shift amount is the same type as the value. Some undefined
    // behavior might occur in the casts below as per [C99 6.5.7.3].
    amt = createIntCast(rewriter, amt, mlirTy, cirAmtTy.isSigned());

    // Lower to the proper arith shift operation.
    if (op.getIsShiftleft())
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(op, mlirTy, val, amt);
    else {
      if (cirValTy.isUnsigned())
        rewriter.replaceOpWithNewOp<mlir::arith::ShRUIOp>(op, mlirTy, val, amt);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ShRSIOp>(op, mlirTy, val, amt);
    }

    return mlir::success();
  }
};

class CIRExp2OpLowering : public mlir::OpConversionPattern<mlir::cir::Exp2Op> {
public:
  using mlir::OpConversionPattern<mlir::cir::Exp2Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::Exp2Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::Exp2Op>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

class CIRSinOpLowering : public mlir::OpConversionPattern<mlir::cir::SinOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::SinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::SinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::SinOp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

template <typename CIROp, typename MLIROp>
class CIRBitOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CIROp op,
                  typename mlir::OpConversionPattern<CIROp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultIntTy = mlir::cast<mlir::IntegerType>(
        this->getTypeConverter()->convertType(op.getType()));
    auto res = rewriter.create<MLIROp>(op->getLoc(), adaptor.getInput());
    auto newOp = createIntCast(rewriter, res->getResult(0), resultIntTy,
                               /*isSigned=*/false);
    rewriter.replaceOp(op, newOp);
    return mlir::LogicalResult::success();
  }
};

using CIRBitClzOpLowering =
    CIRBitOpLowering<mlir::cir::BitClzOp, mlir::math::CountLeadingZerosOp>;
using CIRBitCtzOpLowering =
    CIRBitOpLowering<mlir::cir::BitCtzOp, mlir::math::CountTrailingZerosOp>;
using CIRBitPopcountOpLowering =
    CIRBitOpLowering<mlir::cir::BitPopcountOp, mlir::math::CtPopOp>;

class CIRBitClrsbOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitClrsbOp> {
public:
  using OpConversionPattern<mlir::cir::BitClrsbOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitClrsbOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto zero = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isNeg = rewriter.create<mlir::arith::CmpIOp>(
        op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::slt),
        adaptor.getInput(), zero);

    auto negOne = getConst(rewriter, op.getLoc(), inputTy, -1);
    auto flipped = rewriter.create<mlir::arith::XOrIOp>(
        op.getLoc(), adaptor.getInput(), negOne);

    auto select = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), isNeg, flipped, adaptor.getInput());

    auto resTy = mlir::cast<mlir::IntegerType>(
        getTypeConverter()->convertType(op.getType()));
    auto clz =
        rewriter.create<mlir::math::CountLeadingZerosOp>(op->getLoc(), select);
    auto newClz = createIntCast(rewriter, clz, resTy);

    auto one = getConst(rewriter, op.getLoc(), resTy, 1);
    auto res = rewriter.create<mlir::arith::SubIOp>(op.getLoc(), newClz, one);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitFfsOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitFfsOp> {
public:
  using OpConversionPattern<mlir::cir::BitFfsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitFfsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto inputTy = adaptor.getInput().getType();
    auto ctz = rewriter.create<mlir::math::CountTrailingZerosOp>(
        op.getLoc(), adaptor.getInput());
    auto newCtz = createIntCast(rewriter, ctz, resTy);

    auto one = getConst(rewriter, op.getLoc(), resTy, 1);
    auto ctzAddOne =
        rewriter.create<mlir::arith::AddIOp>(op.getLoc(), newCtz, one);

    auto zeroInputTy = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isZero = rewriter.create<mlir::arith::CmpIOp>(
        op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::eq),
        adaptor.getInput(), zeroInputTy);

    auto zeroResTy = getConst(rewriter, op.getLoc(), resTy, 0);
    auto res = rewriter.create<mlir::arith::SelectOp>(op.getLoc(), isZero,
                                                      zeroResTy, ctzAddOne);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitParityOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BitParityOp> {
public:
  using OpConversionPattern<mlir::cir::BitParityOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BitParityOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto count =
        rewriter.create<mlir::math::CtPopOp>(op.getLoc(), adaptor.getInput());
    auto countMod2 = rewriter.create<mlir::arith::AndIOp>(
        op.getLoc(), count,
        getConst(rewriter, op.getLoc(), count.getType(), 1));
    auto res = createIntCast(rewriter, countMod2, resTy);
    rewriter.replaceOp(op, res);
    return mlir::LogicalResult::success();
  }
};

class CIRConstantOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ConstantOp> {
public:
  using OpConversionPattern<mlir::cir::ConstantOp>::OpConversionPattern;

private:
  // This code is in a separate function rather than part of matchAndRewrite
  // because it is recursive.  There is currently only one level of recursion;
  // when lowing a vector attribute the attributes for the elements also need
  // to be lowered.
  mlir::TypedAttr
  lowerCirAttrToMlirAttr(mlir::Attribute cirAttr,
                         mlir::ConversionPatternRewriter &rewriter) const {
    assert(mlir::isa<mlir::TypedAttr>(cirAttr) &&
           "Can't lower a non-typed attribute");
    auto mlirType = getTypeConverter()->convertType(
        mlir::cast<mlir::TypedAttr>(cirAttr).getType());
    if (auto vecAttr = mlir::dyn_cast<mlir::cir::ConstVectorAttr>(cirAttr)) {
      assert(mlir::isa<mlir::VectorType>(mlirType) &&
             "MLIR type for CIR vector attribute is not mlir::VectorType");
      assert(mlir::isa<mlir::ShapedType>(mlirType) &&
             "mlir::VectorType is not a mlir::ShapedType ??");
      SmallVector<mlir::Attribute> mlirValues;
      for (auto elementAttr : vecAttr.getElts()) {
        mlirValues.push_back(
            this->lowerCirAttrToMlirAttr(elementAttr, rewriter));
      }
      return mlir::DenseElementsAttr::get(
          mlir::cast<mlir::ShapedType>(mlirType), mlirValues);
    } else if (auto boolAttr = mlir::dyn_cast<mlir::cir::BoolAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, boolAttr.getValue());
    } else if (auto floatAttr = mlir::dyn_cast<mlir::cir::FPAttr>(cirAttr)) {
      return rewriter.getFloatAttr(mlirType, floatAttr.getValue());
    } else if (auto intAttr = mlir::dyn_cast<mlir::cir::IntAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, intAttr.getValue());
    } else {
      llvm_unreachable("NYI: unsupported attribute kind lowering to MLIR");
      return {};
    }
  }

public:
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, getTypeConverter()->convertType(op.getType()),
        this->lowerCirAttrToMlirAttr(op.getValue(), rewriter));
    return mlir::LogicalResult::success();
  }
};

class CIRFuncOpLowering : public mlir::OpConversionPattern<mlir::cir::FuncOp> {
public:
  using OpConversionPattern<mlir::cir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto fnType = op.getFunctionType();
    mlir::TypeConverter::SignatureConversion signatureConversion(
        fnType.getNumInputs());

    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = typeConverter->convertType(argType.value());
      if (!convertedType)
        return mlir::failure();
      signatureConversion.addInputs(argType.index(), convertedType);
    }

    mlir::Type resultType =
        getTypeConverter()->convertType(fnType.getReturnType());
    auto fn = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getName(),
        rewriter.getFunctionType(signatureConversion.getConvertedTypes(),
                                 resultType ? mlir::TypeRange(resultType)
                                            : mlir::TypeRange()));

    if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                           &signatureConversion)))
      return mlir::failure();
    rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());

    rewriter.eraseOp(op);
    return mlir::LogicalResult::success();
  }
};

class CIRUnaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::UnaryOp> {
public:
  using OpConversionPattern<mlir::cir::UnaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::UnaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto type = getTypeConverter()->convertType(op.getType());

    switch (op.getKind()) {
    case mlir::cir::UnaryOpKind::Inc: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, type, input, One);
      break;
    }
    case mlir::cir::UnaryOpKind::Dec: {
      auto One = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 1));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, input, One);
      break;
    }
    case mlir::cir::UnaryOpKind::Plus: {
      rewriter.replaceOp(op, op.getInput());
      break;
    }
    case mlir::cir::UnaryOpKind::Minus: {
      auto Zero = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, 0));
      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, Zero, input);
      break;
    }
    case mlir::cir::UnaryOpKind::Not: {
      auto MinusOne = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), type, mlir::IntegerAttr::get(type, -1));
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, type, MinusOne,
                                                       input);
      break;
    }
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBinOpLowering : public mlir::OpConversionPattern<mlir::cir::BinOp> {
public:
  using OpConversionPattern<mlir::cir::BinOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert((adaptor.getLhs().getType() == adaptor.getRhs().getType()) &&
           "inconsistent operands' types not supported yet");
    mlir::Type mlirType = getTypeConverter()->convertType(op.getType());
    assert((mlir::isa<mlir::IntegerType>(mlirType) ||
            mlir::isa<mlir::FloatType>(mlirType) ||
            mlir::isa<mlir::VectorType>(mlirType)) &&
           "operand type not supported yet");

    auto type = op.getLhs().getType();
    if (auto VecType = mlir::dyn_cast<mlir::cir::VectorType>(type)) {
      type = VecType.getEltType();
    }

    switch (op.getKind()) {
    case mlir::cir::BinOpKind::Add:
      if (mlir::isa<mlir::cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Sub:
      if (mlir::isa<mlir::cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Mul:
      if (mlir::isa<mlir::cir::IntType>(type))
        rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Div:
      if (auto ty = mlir::dyn_cast<mlir::cir::IntType>(type)) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::arith::DivUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Rem:
      if (auto ty = mlir::dyn_cast<mlir::cir::IntType>(type)) {
        if (ty.isUnsigned())
          rewriter.replaceOpWithNewOp<mlir::arith::RemUIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
        else
          rewriter.replaceOpWithNewOp<mlir::arith::RemSIOp>(
              op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      } else
        rewriter.replaceOpWithNewOp<mlir::arith::RemFOp>(
            op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::And:
      rewriter.replaceOpWithNewOp<mlir::arith::AndIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Or:
      rewriter.replaceOpWithNewOp<mlir::arith::OrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    case mlir::cir::BinOpKind::Xor:
      rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(
          op, mlirType, adaptor.getLhs(), adaptor.getRhs());
      break;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<mlir::cir::CmpOp> {
public:
  using OpConversionPattern<mlir::cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getLhs().getType();

    mlir::Value mlirResult;

    if (auto ty = mlir::dyn_cast<mlir::cir::IntType>(type)) {
      auto kind = convertCmpKindToCmpIPredicate(op.getKind(), ty.isSigned());
      mlirResult = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ty = mlir::dyn_cast<mlir::cir::CIRFPTypeInterface>(type)) {
      auto kind = convertCmpKindToCmpFPredicate(op.getKind());
      mlirResult = rewriter.create<mlir::arith::CmpFOp>(
          op.getLoc(), kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ty = mlir::dyn_cast<mlir::cir::PointerType>(type)) {
      llvm_unreachable("pointer comparison not supported yet");
    } else {
      return op.emitError() << "unsupported type for CmpOp: " << type;
    }

    // MLIR comparison ops return i1, but cir::CmpOp returns the same type as
    // the LHS value. Since this return value can be used later, we need to
    // restore the type with the extension below.
    auto mlirResultTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, mlirResultTy,
                                                      mlirResult);

    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpRewritePattern<mlir::cir::BrOp> {
public:
  using OpRewritePattern<mlir::cir::BrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest());
    return mlir::LogicalResult::success();
  }
};

class CIRScopeOpLowering
    : public mlir::OpConversionPattern<mlir::cir::ScopeOp> {
  using mlir::OpConversionPattern<mlir::cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::ScopeOp scopeOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Empty scope: just remove it.
    if (scopeOp.getRegion().empty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    for (auto &block : scopeOp.getRegion()) {
      rewriter.setInsertionPointToEnd(&block);
      auto *terminator = block.getTerminator();
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(
          terminator, terminator->getOperands());
    }

    SmallVector<mlir::Type> mlirResultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(scopeOp->getResultTypes(),
                                                      mlirResultTypes)))
      return mlir::LogicalResult::failure();
    rewriter.setInsertionPoint(scopeOp);
    auto newScopeOp = rewriter.create<mlir::memref::AllocaScopeOp>(
        scopeOp.getLoc(), mlirResultTypes);
    rewriter.inlineRegionBefore(scopeOp.getScopeRegion(),
                                newScopeOp.getBodyRegion(),
                                newScopeOp.getBodyRegion().end());
    rewriter.replaceOp(scopeOp, newScopeOp);

    return mlir::LogicalResult::success();
  }
};

struct CIRBrCondOpLowering
    : public mlir::OpConversionPattern<mlir::cir::BrCondOp> {
  using mlir::OpConversionPattern<mlir::cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::arith::TruncIOp>(
        brOp.getLoc(), rewriter.getI1Type(), condition);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        brOp, i1Condition.getResult(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRTernaryOpLowering
    : public mlir::OpConversionPattern<mlir::cir::TernaryOp> {
public:
  using OpConversionPattern<mlir::cir::TernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::TernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto condition = adaptor.getCond();
    auto i1Condition = rewriter.create<mlir::arith::TruncIOp>(
        op.getLoc(), rewriter.getI1Type(), condition);
    SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return mlir::failure();

    auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), resultTypes,
                                                 i1Condition.getResult(), true);
    auto *thenBlock = &ifOp.getThenRegion().front();
    auto *elseBlock = &ifOp.getElseRegion().front();
    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), thenBlock,
                               thenBlock->end());
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), elseBlock,
                               elseBlock->end());

    rewriter.replaceOp(op, ifOp);
    return mlir::success();
  }
};

class CIRYieldOpLowering
    : public mlir::OpConversionPattern<mlir::cir::YieldOp> {
public:
  using OpConversionPattern<mlir::cir::YieldOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

class CIRIfOpLowering : public mlir::OpConversionPattern<mlir::cir::IfOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::IfOp ifop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto condition = adaptor.getCondition();
    auto i1Condition = rewriter.create<mlir::arith::TruncIOp>(
        ifop->getLoc(), rewriter.getI1Type(), condition);
    auto newIfOp = rewriter.create<mlir::scf::IfOp>(
        ifop->getLoc(), ifop->getResultTypes(), i1Condition);
    auto *thenBlock = rewriter.createBlock(&newIfOp.getThenRegion());
    rewriter.inlineBlockBefore(&ifop.getThenRegion().front(), thenBlock,
                               thenBlock->end());
    if (!ifop.getElseRegion().empty()) {
      auto *elseBlock = rewriter.createBlock(&newIfOp.getElseRegion());
      rewriter.inlineBlockBefore(&ifop.getElseRegion().front(), elseBlock,
                                 elseBlock->end());
    }
    rewriter.replaceOp(ifop, newIfOp);
    return mlir::success();
  }
};

class CIRGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GlobalOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::OpBuilder b(moduleOp.getContext());

    const auto CIRSymType = op.getSymType();
    auto convertedType = getTypeConverter()->convertType(CIRSymType);
    if (!convertedType)
      return mlir::failure();
    auto memrefType = dyn_cast<mlir::MemRefType>(convertedType);
    if (!memrefType)
      memrefType = mlir::MemRefType::get({}, convertedType);
    // Add an optional alignment to the global memref.
    mlir::IntegerAttr memrefAlignment =
        op.getAlignment()
            ? mlir::IntegerAttr::get(b.getI64Type(), op.getAlignment().value())
            : mlir::IntegerAttr();
    // Add an optional initial value to the global memref.
    mlir::Attribute initialValue = mlir::Attribute();
    std::optional<mlir::Attribute> init = op.getInitialValue();
    if (init.has_value()) {
      if (auto constArr = mlir::dyn_cast<mlir::cir::ZeroAttr>(init.value())) {
        if (memrefType.getShape().size()) {
          auto elementType = memrefType.getElementType();
          auto rtt =
              mlir::RankedTensorType::get(memrefType.getShape(), elementType);
          if (mlir::isa<mlir::IntegerType>(elementType))
            initialValue = mlir::DenseIntElementsAttr::get(rtt, 0);
          else if (mlir::isa<mlir::FloatType>(elementType)) {
            auto floatZero = mlir::FloatAttr::get(elementType, 0.0).getValue();
            initialValue = mlir::DenseFPElementsAttr::get(rtt, floatZero);
          } else
            llvm_unreachable("GlobalOp lowering unsuppored element type");
        } else {
          auto rtt = mlir::RankedTensorType::get({}, convertedType);
          if (mlir::isa<mlir::IntegerType>(convertedType))
            initialValue = mlir::DenseIntElementsAttr::get(rtt, 0);
          else if (mlir::isa<mlir::FloatType>(convertedType)) {
            auto floatZero =
                mlir::FloatAttr::get(convertedType, 0.0).getValue();
            initialValue = mlir::DenseFPElementsAttr::get(rtt, floatZero);
          } else
            llvm_unreachable("GlobalOp lowering unsuppored type");
        }
      } else if (auto intAttr =
                     mlir::dyn_cast<mlir::cir::IntAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({}, convertedType);
        initialValue = mlir::DenseIntElementsAttr::get(rtt, intAttr.getValue());
      } else if (auto fltAttr =
                     mlir::dyn_cast<mlir::cir::FPAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({}, convertedType);
        initialValue = mlir::DenseFPElementsAttr::get(rtt, fltAttr.getValue());
      } else if (auto boolAttr =
                     mlir::dyn_cast<mlir::cir::BoolAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({}, convertedType);
        initialValue =
            mlir::DenseIntElementsAttr::get(rtt, (char)boolAttr.getValue());
      } else
        llvm_unreachable(
            "GlobalOp lowering with initial value is not fully supported yet");
    }

    // Add symbol visibility
    std::string sym_visibility = op.isPrivate() ? "private" : "public";

    rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
        op, b.getStringAttr(op.getSymName()),
        /*sym_visibility=*/b.getStringAttr(sym_visibility),
        /*type=*/memrefType, initialValue,
        /*constant=*/op.getConstant(),
        /*alignment=*/memrefAlignment);

    return mlir::success();
  }
};

class CIRGetGlobalOpLowering
    : public mlir::OpConversionPattern<mlir::cir::GetGlobalOp> {
public:
  using OpConversionPattern<mlir::cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
    // CIRGen should mitigate this and not emit the get_global.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    auto type = getTypeConverter()->convertType(op.getType());
    auto symbol = op.getName();
    rewriter.replaceOpWithNewOp<mlir::memref::GetGlobalOp>(op, type, symbol);
    return mlir::success();
  }
};

class CIRVectorCreateLowering
    : public mlir::OpConversionPattern<mlir::cir::VecCreateOp> {
public:
  using OpConversionPattern<mlir::cir::VecCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vecTy = mlir::dyn_cast<mlir::cir::VectorType>(op.getType());
    assert(vecTy && "result type of cir.vec.create op is not VectorType");
    auto elementTy = typeConverter->convertType(vecTy.getEltType());
    auto loc = op.getLoc();
    auto zeroElement = rewriter.getZeroAttr(elementTy);
    mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(
        loc,
        mlir::DenseElementsAttr::get(
            mlir::VectorType::get(vecTy.getSize(), elementTy), zeroElement));
    assert(vecTy.getSize() == op.getElements().size() &&
           "cir.vec.create op count doesn't match vector type elements count");
    for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
      mlir::Value indexValue =
          getConst(rewriter, loc, rewriter.getI64Type(), i);
      result = rewriter.create<mlir::vector::InsertElementOp>(
          loc, adaptor.getElements()[i], result, indexValue);
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class CIRVectorInsertLowering
    : public mlir::OpConversionPattern<mlir::cir::VecInsertOp> {
public:
  using OpConversionPattern<mlir::cir::VecInsertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecInsertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::vector::InsertElementOp>(
        op, adaptor.getValue(), adaptor.getVec(), adaptor.getIndex());
    return mlir::success();
  }
};

class CIRVectorExtractLowering
    : public mlir::OpConversionPattern<mlir::cir::VecExtractOp> {
public:
  using OpConversionPattern<mlir::cir::VecExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecExtractOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractElementOp>(
        op, adaptor.getVec(), adaptor.getIndex());
    return mlir::success();
  }
};

class CIRVectorCmpOpLowering
    : public mlir::OpConversionPattern<mlir::cir::VecCmpOp> {
public:
  using OpConversionPattern<mlir::cir::VecCmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::VecCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(mlir::isa<mlir::cir::VectorType>(op.getType()) &&
           mlir::isa<mlir::cir::VectorType>(op.getLhs().getType()) &&
           mlir::isa<mlir::cir::VectorType>(op.getRhs().getType()) &&
           "Vector compare with non-vector type");
    auto elementType =
        mlir::cast<mlir::cir::VectorType>(op.getLhs().getType()).getEltType();
    mlir::Value bitResult;
    if (auto intType = mlir::dyn_cast<mlir::cir::IntType>(elementType)) {
      bitResult = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(),
          convertCmpKindToCmpIPredicate(op.getKind(), intType.isSigned()),
          adaptor.getLhs(), adaptor.getRhs());
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(elementType)) {
      bitResult = rewriter.create<mlir::arith::CmpFOp>(
          op.getLoc(), convertCmpKindToCmpFPredicate(op.getKind()),
          adaptor.getLhs(), adaptor.getRhs());
    } else {
      return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(
        op, typeConverter->convertType(op.getType()), bitResult);
    return mlir::success();
  }
};

class CIRCastOpLowering : public mlir::OpConversionPattern<mlir::cir::CastOp> {
public:
  using OpConversionPattern<mlir::cir::CastOp>::OpConversionPattern;

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::cir::CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isa<mlir::cir::VectorType>(op.getSrc().getType()))
      llvm_unreachable("CastOp lowering for vector type is not supported yet");
    auto src = adaptor.getSrc();
    auto dstType = op.getResult().getType();
    using CIR = mlir::cir::CastKind;
    switch (op.getKind()) {
    case CIR::array_to_ptrdecay: {
      auto newDstType = mlir::cast<mlir::MemRefType>(convertTy(dstType));
      rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
          op, newDstType, src, 0, std::nullopt, std::nullopt);
      return mlir::success();
    }
    case CIR::int_to_bool: {
      auto zero = rewriter.create<mlir::cir::ConstantOp>(
          src.getLoc(), op.getSrc().getType(),
          mlir::cir::IntAttr::get(op.getSrc().getType(), 0));
      rewriter.replaceOpWithNewOp<mlir::cir::CmpOp>(
          op, mlir::cir::BoolType::get(getContext()), mlir::cir::CmpOpKind::ne,
          op.getSrc(), zero);
      return mlir::success();
    }
    case CIR::integral: {
      auto newDstType = convertTy(dstType);
      auto srcType = op.getSrc().getType();
      mlir::cir::IntType srcIntType = mlir::cast<mlir::cir::IntType>(srcType);
      auto newOp =
          createIntCast(rewriter, src, newDstType, srcIntType.isSigned());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::floating: {
      auto newDstType = convertTy(dstType);
      auto srcTy = op.getSrc().getType();
      auto dstTy = op.getResult().getType();

      if (!mlir::isa<mlir::cir::CIRFPTypeInterface>(dstTy) ||
          !mlir::isa<mlir::cir::CIRFPTypeInterface>(srcTy))
        return op.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

      auto getFloatWidth = [](mlir::Type ty) -> unsigned {
        return mlir::cast<mlir::cir::CIRFPTypeInterface>(ty).getWidth();
      };

      if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
        rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_bool: {
      auto dstTy = mlir::cast<mlir::cir::BoolType>(op.getType());
      auto newDstType = convertTy(dstTy);
      auto kind = mlir::arith::CmpFPredicate::UNE;

      // Check if float is not equal to zero.
      auto zeroFloat = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), src.getType(), mlir::FloatAttr::get(src.getType(), 0.0));

      // Extend comparison result to either bool (C++) or int (C).
      mlir::Value cmpResult = rewriter.create<mlir::arith::CmpFOp>(
          op.getLoc(), kind, src, zeroFloat);
      rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, newDstType,
                                                        cmpResult);
      return mlir::success();
    }
    case CIR::bool_to_int: {
      auto dstTy = mlir::cast<mlir::cir::IntType>(op.getType());
      auto newDstType = mlir::cast<mlir::IntegerType>(convertTy(dstTy));
      auto newOp = createIntCast(rewriter, src, newDstType);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::bool_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::int_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<mlir::cir::IntType>(op.getSrc().getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_int: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<mlir::cir::IntType>(op.getResult().getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::FPToUIOp>(op, newDstType, src);
      return mlir::success();
    }
    default:
      break;
    }
    return mlir::failure();
  }
};

class CIRPtrStrideOpLowering
    : public mlir::OpConversionPattern<mlir::cir::PtrStrideOp> {
public:
  using mlir::OpConversionPattern<mlir::cir::PtrStrideOp>::OpConversionPattern;

  // Return true if PtrStrideOp is produced by cast with array_to_ptrdecay kind
  // and they are in the same block.
  inline bool isCastArrayToPtrConsumer(mlir::cir::PtrStrideOp op) const {
    auto defOp = op->getOperand(0).getDefiningOp();
    if (!defOp)
      return false;
    auto castOp = dyn_cast<mlir::cir::CastOp>(defOp);
    if (!castOp)
      return false;
    if (castOp.getKind() != mlir::cir::CastKind::array_to_ptrdecay)
      return false;
    if (!castOp->hasOneUse())
      return false;
    if (!castOp->isBeforeInBlock(op))
      return false;
    return true;
  }

  // Return true if all the PtrStrideOp users are load, store or cast
  // with array_to_ptrdecay kind and they are in the same block.
  inline bool
  isLoadStoreOrCastArrayToPtrProduer(mlir::cir::PtrStrideOp op) const {
    if (op.use_empty())
      return false;
    for (auto *user : op->getUsers()) {
      if (!op->isBeforeInBlock(user))
        return false;
      if (isa<mlir::cir::LoadOp>(*user) || isa<mlir::cir::StoreOp>(*user))
        continue;
      auto castOp = dyn_cast<mlir::cir::CastOp>(*user);
      if (castOp &&
          (castOp.getKind() == mlir::cir::CastKind::array_to_ptrdecay))
        continue;
      return false;
    }
    return true;
  }

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  // Rewrite
  //        %0 = cir.cast(array_to_ptrdecay, %base)
  //        cir.ptr_stride(%0, %stride)
  // to
  //        memref.reinterpret_cast (%base, %stride)
  //
  // MemRef Dialect doesn't have GEP-like operation. memref.reinterpret_cast
  // only been used to propogate %base and %stride to memref.load/store and
  // should be erased after the conversion.
  mlir::LogicalResult
  matchAndRewrite(mlir::cir::PtrStrideOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!isCastArrayToPtrConsumer(op))
      return mlir::failure();
    if (!isLoadStoreOrCastArrayToPtrProduer(op))
      return mlir::failure();
    auto baseOp = adaptor.getBase().getDefiningOp();
    if (!baseOp)
      return mlir::failure();
    if (!isa<mlir::memref::ReinterpretCastOp>(baseOp))
      return mlir::failure();
    auto base = baseOp->getOperand(0);
    auto dstType = op.getResult().getType();
    auto newDstType = mlir::cast<mlir::MemRefType>(convertTy(dstType));
    auto stride = adaptor.getStride();
    auto indexType = rewriter.getIndexType();
    // Generate casting if the stride is not index type.
    if (stride.getType() != indexType)
      stride = rewriter.create<mlir::arith::IndexCastOp>(op.getLoc(), indexType,
                                                         stride);
    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, newDstType, base, stride, std::nullopt, std::nullopt);
    rewriter.eraseOp(baseOp);
    return mlir::success();
  }
};

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering, CIRBrOpLowering>(patterns.getContext());

  patterns.add<
      CIRCmpOpLowering, CIRCallOpLowering, CIRUnaryOpLowering, CIRBinOpLowering,
      CIRLoadOpLowering, CIRConstantOpLowering, CIRStoreOpLowering,
      CIRAllocaOpLowering, CIRFuncOpLowering, CIRScopeOpLowering,
      CIRBrCondOpLowering, CIRTernaryOpLowering, CIRYieldOpLowering,
      CIRCosOpLowering, CIRGlobalOpLowering, CIRGetGlobalOpLowering,
      CIRCastOpLowering, CIRPtrStrideOpLowering, CIRSqrtOpLowering,
      CIRCeilOpLowering, CIRExp2OpLowering, CIRExpOpLowering, CIRFAbsOpLowering,
      CIRFloorOpLowering, CIRLog10OpLowering, CIRLog2OpLowering,
      CIRLogOpLowering, CIRRoundOpLowering, CIRPtrStrideOpLowering,
      CIRSinOpLowering, CIRShiftOpLowering, CIRBitClzOpLowering,
      CIRBitCtzOpLowering, CIRBitPopcountOpLowering, CIRBitClrsbOpLowering,
      CIRBitFfsOpLowering, CIRBitParityOpLowering, CIRIfOpLowering,
      CIRVectorCreateLowering, CIRVectorInsertLowering,
      CIRVectorExtractLowering, CIRVectorCmpOpLowering>(converter,
                                                        patterns.getContext());
}

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](mlir::cir::PointerType type) -> mlir::Type {
    auto ty = converter.convertType(type.getPointee());
    // FIXME: The pointee type might not be converted (e.g. struct)
    if (!ty)
      return nullptr;
    if (isa<mlir::cir::ArrayType>(type.getPointee()))
      return ty;
    return mlir::MemRefType::get({}, ty);
  });
  converter.addConversion(
      [&](mlir::IntegerType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::FloatType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::cir::VoidType type) -> mlir::Type { return {}; });
  converter.addConversion([&](mlir::cir::IntType type) -> mlir::Type {
    // arith dialect ops doesn't take signed integer -- drop cir sign here
    return mlir::IntegerType::get(
        type.getContext(), type.getWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  });
  converter.addConversion([&](mlir::cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 8);
  });
  converter.addConversion([&](mlir::cir::SingleType type) -> mlir::Type {
    return mlir::FloatType::getF32(type.getContext());
  });
  converter.addConversion([&](mlir::cir::DoubleType type) -> mlir::Type {
    return mlir::FloatType::getF64(type.getContext());
  });
  converter.addConversion([&](mlir::cir::FP80Type type) -> mlir::Type {
    return mlir::FloatType::getF80(type.getContext());
  });
  converter.addConversion([&](mlir::cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](mlir::cir::ArrayType type) -> mlir::Type {
    SmallVector<int64_t> shape;
    mlir::Type curType = type;
    while (auto arrayType = dyn_cast<mlir::cir::ArrayType>(curType)) {
      shape.push_back(arrayType.getSize());
      curType = arrayType.getEltType();
    }
    auto elementType = converter.convertType(curType);
    // FIXME: The element type might not be converted (e.g. struct)
    if (!elementType)
      return nullptr;
    return mlir::MemRefType::get(shape, elementType);
  });
  converter.addConversion([&](mlir::cir::VectorType type) -> mlir::Type {
    auto ty = converter.convertType(type.getEltType());
    return mlir::VectorType::get(type.getSize(), ty);
  });

  return converter;
}

void ConvertCIRToMLIRPass::runOnOperation() {
  auto module = getOperation();

  auto converter = prepareTypeConverter();

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRLoopToSCFConversionPatterns(patterns, converter);
  populateCIRToMLIRConversionPatterns(patterns, converter);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target
      .addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                       mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                       mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                       mlir::math::MathDialect, mlir::vector::VectorDialect>();
  target.addIllegalDialect<mlir::cir::CIRDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<llvm::Module>
lowerFromCIRToMLIRToLLVMIR(mlir::ModuleOp theModule,
                           std::unique_ptr<mlir::MLIRContext> mlirCtx,
                           LLVMContext &llvmCtx) {
  mlir::PassManager pm(mlirCtx.get());

  pm.addPass(createConvertCIRToMLIRPass());
  pm.addPass(createConvertMLIRToLLVMPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  mlir::registerBuiltinDialectTranslation(*mlirCtx);
  mlir::registerLLVMDialectTranslation(*mlirCtx);
  mlir::registerOpenMPDialectTranslation(*mlirCtx);

  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, llvmCtx);

  if (!llvmModule)
    report_fatal_error("Lowering from LLVMIR dialect to llvm IR failed!");

  return llvmModule;
}

std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass() {
  return std::make_unique<ConvertCIRToMLIRPass>();
}

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp theModule,
                                  mlir::MLIRContext *mlirCtx) {
  mlir::PassManager pm(mlirCtx);

  pm.addPass(createConvertCIRToMLIRPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to LLVMIR dialect!");

  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error("Verification of the final LLVMIR dialect failed!");

  return theModule;
}

} // namespace cir
