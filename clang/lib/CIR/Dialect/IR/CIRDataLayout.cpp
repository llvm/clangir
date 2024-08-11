#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/MissingFeatures.h"

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

using namespace cir;

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} { reset(); }

void CIRDataLayout::reset() {
  clear();

  // NOTE(cir): Alignment setter functions are skipped as these should already
  // be set in MLIR's data layout.
}

void CIRDataLayout::clear() {}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
llvm::Align CIRDataLayout::getAlignment(mlir::Type Ty, bool abi_or_pref) const {

  // FIXME(cir): This does not account for differnt address spaces, and relies
  // on CIR's data layout to give the proper alignment.
  assert(!::cir::MissingFeatures::addressSpace());

  // Fetch type alignment from MLIR's data layout.
  uint align = abi_or_pref ? layout.getTypeABIAlignment(Ty)
                           : layout.getTypePreferredAlignment(Ty);
  return llvm::Align(align);
}

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
llvm::TypeSize CIRDataLayout::getTypeSizeInBits(mlir::Type Ty) const {
  assert(!::cir::MissingFeatures::typeIsSized() &&
         "Cannot getTypeInfo() on a type that is unsized!");

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!::cir::MissingFeatures::addressSpace());

  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(Ty));
}
