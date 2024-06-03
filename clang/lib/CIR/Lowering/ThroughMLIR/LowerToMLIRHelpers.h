#ifndef LLVM_CLANG_LIB_CIR_LOWERING_THROUGHMLIR_LOWERTOMLIRHELPERS_H
#define LLVM_CLANG_LIB_CIR_LOWERING_THROUGHMLIR_LOWERTOMLIRHELPERS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

#include <cassert>

namespace cir {

template <typename T>
inline mlir::Value getConst(mlir::ConversionPatternRewriter &rewriter,
                            mlir::Location loc, mlir::Type ty, T value) {
  assert((mlir::isa<mlir::IntegerType>(ty) || mlir::isa<mlir::FloatType>(ty)) &&
         "expected integer or floating-point type");

  if (mlir::isa<mlir::IntegerType>(ty))
    return rewriter.create<mlir::arith::ConstantOp>(
        loc, ty, mlir::IntegerAttr::get(ty, value));

  return rewriter.create<mlir::arith::ConstantOp>(
      loc, ty, mlir::FloatAttr::get(ty, value));
}

inline mlir::Value createIntCast(mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Value src, mlir::Type dstTy,
                                 bool isSigned = false) {
  auto srcTy = src.getType();
  assert(mlir::isa<mlir::IntegerType>(srcTy) && "expected integer source type");
  assert(mlir::isa<mlir::IntegerType>(dstTy) && "expected integer dest type");

  auto srcWidth = llvm::cast<mlir::IntegerType>(srcTy).getWidth();
  auto dstWidth = llvm::cast<mlir::IntegerType>(dstTy).getWidth();
  auto loc = src.getLoc();

  if (dstWidth > srcWidth && isSigned)
    return rewriter.create<mlir::arith::ExtSIOp>(loc, dstTy, src);
  if (dstWidth > srcWidth)
    return rewriter.create<mlir::arith::ExtUIOp>(loc, dstTy, src);
  if (dstWidth < srcWidth)
    return rewriter.create<mlir::arith::TruncIOp>(loc, dstTy, src);
  return rewriter.create<mlir::arith::BitcastOp>(loc, dstTy, src);
}

} // namespace cir

#endif
