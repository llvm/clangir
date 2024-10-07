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

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/TypeSize.h"

namespace cir {

class StructLayout;

// FIXME(cir): This might be replaced by a CIRDataLayout interface which can
// provide the same functionalities.
class CIRDataLayout {
  bool bigEndian = false;

  /// Primitive type alignment data. This is sorted by type and bit
  /// width during construction.
  llvm::DataLayout::PrimitiveSpec structAlignment;

  // The StructType -> StructLayout map.
  mutable void *layoutMap = nullptr;

public:
  mlir::DataLayout layout;

  /// Constructs a DataLayout the module's data layout attribute.
  CIRDataLayout(mlir::ModuleOp modOp);

  /// Parse a data layout string (with fallback to default values).
  void reset(mlir::DataLayoutSpecInterface spec);

  // Free all internal data structures.
  void clear();

  bool isBigEndian() const { return bigEndian; }

  /// Returns a StructLayout object, indicating the alignment of the
  /// struct, its size, and the offsets of its fields.
  ///
  /// Note that this information is lazily cached.
  const StructLayout *getStructLayout(mlir::cir::StructType ty) const;

  /// Internal helper method that returns requested alignment for type.
  llvm::Align getAlignment(mlir::Type ty, bool abiOrPref) const;

  llvm::Align getABITypeAlign(mlir::Type ty) const {
    return getAlignment(ty, true);
  }

  llvm::Align getPrefTypeAlign(mlir::Type ty) const {
    return getAlignment(ty, false);
  }

  /// Returns the maximum number of bytes that may be overwritten by
  /// storing the specified type.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// For example, returns 5 for i36 and 10 for x86_fp80.
  llvm::TypeSize getTypeStoreSize(mlir::Type ty) const {
    llvm::TypeSize baseSize = getTypeSizeInBits(ty);
    return {llvm::divideCeil(baseSize.getKnownMinValue(), 8),
            baseSize.isScalable()};
  }

  /// Returns the offset in bytes between successive objects of the
  /// specified type, including alignment padding.
  ///
  /// If Ty is a scalable vector type, the scalable property will be set and
  /// the runtime size will be a positive integer multiple of the base size.
  ///
  /// This is the amount that alloca reserves for this type. For example,
  /// returns 12 or 16 for x86_fp80, depending on alignment.
  llvm::TypeSize getTypeAllocSize(mlir::Type ty) const {
    // Round up to the next alignment boundary.
    return llvm::alignTo(getTypeStoreSize(ty), getABITypeAlign(ty).value());
  }

  llvm::TypeSize getPointerTypeSizeInBits(mlir::Type ty) const {
    assert(mlir::isa<mlir::cir::PointerType>(ty) &&
           "This should only be called with a pointer type");
    return layout.getTypeSizeInBits(ty);
  }

  llvm::TypeSize getTypeSizeInBits(mlir::Type ty) const;

  mlir::Type getIntPtrType(mlir::Type ty) const {
    assert(mlir::isa<mlir::cir::PointerType>(ty) && "Expected pointer type");
    auto intTy = mlir::cir::IntType::get(ty.getContext(),
                                         getPointerTypeSizeInBits(ty), false);
    return intTy;
  }
};

/// Used to lazily calculate structure layout information for a target machine,
/// based on the DataLayout structure.
class StructLayout final
    : public llvm::TrailingObjects<StructLayout, llvm::TypeSize> {
  llvm::TypeSize structSize;
  llvm::Align structAlignment;
  unsigned isPadded : 1;
  unsigned numElements : 31;

public:
  llvm::TypeSize getSizeInBytes() const { return structSize; }

  llvm::TypeSize getSizeInBits() const { return 8 * structSize; }

  llvm::Align getAlignment() const { return structAlignment; }

  /// Returns whether the struct has padding or not between its fields.
  /// NB: Padding in nested element is not taken into account.
  bool hasPadding() const { return isPadded; }

  /// Given a valid byte offset into the structure, returns the structure
  /// index that contains it.
  unsigned getElementContainingOffset(uint64_t fixedOffset) const;

  llvm::MutableArrayRef<llvm::TypeSize> getMemberOffsets() {
    return llvm::MutableArrayRef(getTrailingObjects<llvm::TypeSize>(),
                                 numElements);
  }

  llvm::ArrayRef<llvm::TypeSize> getMemberOffsets() const {
    return llvm::ArrayRef(getTrailingObjects<llvm::TypeSize>(), numElements);
  }

  llvm::TypeSize getElementOffset(unsigned idx) const {
    assert(idx < numElements && "Invalid element idx!");
    return getMemberOffsets()[idx];
  }

  llvm::TypeSize getElementOffsetInBits(unsigned idx) const {
    return getElementOffset(idx) * 8;
  }

private:
  friend class CIRDataLayout; // Only DataLayout can create this class

  StructLayout(mlir::cir::StructType st, const CIRDataLayout &dl);

  size_t numTrailingObjects(OverloadToken<llvm::TypeSize>) const {
    return numElements;
  }
};

} // namespace cir

#endif
