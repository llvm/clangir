//===--- CIRDataLayout.h - CIR Data Layout Information ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Provides a LLVM-like API wrapper to DLTI and MLIR layout queries. This makes
// it easier to port some of LLVM codegen layout logic to CIR.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H

#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/TypeSize.h"

namespace cir {

// FIXME(cir): This might be replaced by a CIRDataLayout interface which can
// provide the same functionalities.
class CIRDataLayout {
  bool bigEndian = false;

public:
  mlir::DataLayout layout;

  /// Constructs a DataLayout the module's data layout attribute.
  CIRDataLayout(mlir::ModuleOp modOp);

  /// Parse a data layout string (with fallback to default values).
  void reset();

  // Free all internal data structures.
  void clear();

  bool isBigEndian() const { return bigEndian; }


  /// Internal helper method that returns requested alignment for type.
  llvm::Align getAlignment(mlir::Type Ty, bool abi_or_pref) const;

  llvm::Align getABITypeAlign(mlir::Type ty) const {
    return getAlignment(ty, true);
  }

  /// Returns the maximum number of bytes that may be overwritten by
  /// storing the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 5 for i36 and 10 for x86_fp80.
  llvm::TypeSize getTypeStoreSize(mlir::Type Ty) const {
    llvm::TypeSize BaseSize = getTypeSizeInBits(Ty);
    return {llvm::divideCeil(BaseSize.getKnownMinValue(), 8),
            BaseSize.isScalable()};
  }

  /// Returns the offset in bytes between successive objects of the
  /// specified type, including alignment padding.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// This is the amount that alloca reserves for this type. For example,
  /// returns 12 or 16 for x86_fp80, depending on alignment.
  llvm::TypeSize getTypeAllocSize(mlir::Type Ty) const {
    // Round up to the next alignment boundary.
    return llvm::alignTo(getTypeStoreSize(Ty), getABITypeAlign(Ty).value());
  }

  llvm::TypeSize getPointerTypeSizeInBits(mlir::Type Ty) const {
    assert(mlir::isa<mlir::cir::PointerType>(Ty) &&
           "This should only be called with a pointer type");
    return layout.getTypeSizeInBits(Ty);
  }

  // The implementation of this method is provided inline as it is particularly
  // well suited to constant folding when called on a specific Type subclass.
  llvm::TypeSize getTypeSizeInBits(mlir::Type Ty) const;

  mlir::Type getIntPtrType(mlir::Type Ty) const {
    assert(mlir::isa<mlir::cir::PointerType>(Ty) && "Expected pointer type");
    auto IntTy = mlir::cir::IntType::get(Ty.getContext(),
                                         getPointerTypeSizeInBits(Ty), false);
    return IntTy;
  }
};

} // namespace cir

#endif
