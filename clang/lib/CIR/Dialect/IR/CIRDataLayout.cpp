#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/IR/DataLayout.h"

using namespace cir;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(mlir::cir::StructType st, const CIRDataLayout &dl)
    : structSize(llvm::TypeSize::getFixed(0)) {
  assert(!st.isIncomplete() && "Cannot get layout of opaque structs");
  isPadded = false;
  numElements = st.getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = numElements; i != e; ++i) {
    mlir::Type ty = st.getMembers()[i];
    if (i == 0 && ::cir::MissingFeatures::typeIsScalableType())
      llvm_unreachable("Scalable types are not yet supported in CIR");

    assert(!::cir::MissingFeatures::recordDeclIsPacked() &&
           "Cannot identify packed structs");
    const llvm::Align tyAlign =
        st.getPacked() ? llvm::Align(1) : dl.getABITypeAlign(ty);

    // Add padding if necessary to align the data element properly.
    // Currently the only structure with scalable size will be the homogeneous
    // scalable vector types. Homogeneous scalable vector types have members of
    // the same data type so no alignment issue will happen. The condition here
    // assumes so and needs to be adjusted if this assumption changes (e.g. we
    // support structures with arbitrary scalable data type, or structure that
    // contains both fixed size and scalable size data type members).
    if (!structSize.isScalable() && !isAligned(tyAlign, structSize)) {
      isPadded = true;
      structSize = llvm::TypeSize::getFixed(alignTo(structSize, tyAlign));
    }

    // Keep track of maximum alignment constraint.
    structAlignment = std::max(tyAlign, structAlignment);

    getMemberOffsets()[i] = structSize;
    // Consume space for this data item
    structSize += dl.getTypeAllocSize(ty);
  }

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!structSize.isScalable() && !isAligned(structAlignment, structSize)) {
    isPadded = true;
    structSize = llvm::TypeSize::getFixed(alignTo(structSize, structAlignment));
  }
}

/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t fixedOffset) const {
  assert(!structSize.isScalable() &&
         "Cannot get element at offset for structure containing scalable "
         "vector types");
  llvm::TypeSize offset = llvm::TypeSize::getFixed(fixedOffset);
  llvm::ArrayRef<llvm::TypeSize> memberOffsets = getMemberOffsets();

  const auto *si =
      std::upper_bound(memberOffsets.begin(), memberOffsets.end(), offset,
                       [](llvm::TypeSize lhs, llvm::TypeSize rhs) -> bool {
                         return llvm::TypeSize::isKnownLT(lhs, rhs);
                       });
  assert(si != memberOffsets.begin() && "Offset not in structure type!");
  --si;
  assert(llvm::TypeSize::isKnownLE(*si, offset) && "upper_bound didn't work");
  assert((si == memberOffsets.begin() ||
          llvm::TypeSize::isKnownLE(*(si - 1), offset)) &&
         (si + 1 == memberOffsets.end() ||
          llvm::TypeSize::isKnownGT(*(si + 1), offset)) &&
         "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return si - memberOffsets.begin();
}

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

namespace {

class StructLayoutMap {
  using LayoutInfoTy = llvm::DenseMap<mlir::cir::StructType, StructLayout *>;
  LayoutInfoTy layoutInfo;

public:
  ~StructLayoutMap() {
    // Remove any layouts.
    for (const auto &i : layoutInfo) {
      StructLayout *value = i.second;
      value->~StructLayout();
      free(value);
    }
  }

  StructLayout *&operator[](mlir::cir::StructType sTy) {
    return layoutInfo[sTy];
  }
};

} // namespace

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} {
  reset(modOp.getDataLayoutSpec());
}

void CIRDataLayout::reset(mlir::DataLayoutSpecInterface spec) {
  clear();

  bigEndian = false;
  if (spec) {
    auto key = mlir::StringAttr::get(
        spec.getContext(), mlir::DLTIDialect::kDataLayoutEndiannessKey);
    if (auto entry = spec.getSpecForIdentifier(key))
      if (auto str = llvm::dyn_cast<mlir::StringAttr>(entry.getValue()))
        bigEndian = str == mlir::DLTIDialect::kDataLayoutEndiannessBig;
  }

  layoutMap = nullptr;

  // ManglingMode = MM_None;
  // NonIntegralAddressSpaces.clear();
  structAlignment =
      llvm::DataLayout::PrimitiveSpec{0, llvm::Align(1), llvm::Align(8)};

  // NOTE(cir): Alignment setter functions are skipped as these should already
  // be set in MLIR's data layout.
}

void CIRDataLayout::clear() {
  delete static_cast<StructLayoutMap *>(layoutMap);
  layoutMap = nullptr;
}

const StructLayout *
CIRDataLayout::getStructLayout(mlir::cir::StructType ty) const {
  if (!layoutMap)
    layoutMap = new StructLayoutMap();

  StructLayoutMap *stm = static_cast<StructLayoutMap *>(layoutMap);
  StructLayout *&sl = (*stm)[ty];
  if (sl)
    return sl;

  // Otherwise, create the struct layout.  Because it is variable length, we
  // malloc it, then use placement new.
  StructLayout *l = (StructLayout *)llvm::safe_malloc(
      StructLayout::totalSizeToAlloc<llvm::TypeSize>(ty.getNumElements()));

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  sl = l;

  new (l) StructLayout(ty, *this);

  return l;
}

/*!
  \param abiOrPref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abiOrPref == true) or preferred alignment (\a abiOrPref
  == false) for the requested type \a Ty.
 */
llvm::Align CIRDataLayout::getAlignment(mlir::Type ty, bool abiOrPref) const {

  if (llvm::isa<mlir::cir::StructType>(ty)) {
    // Packed structure types always have an ABI alignment of one.
    if (::cir::MissingFeatures::recordDeclIsPacked() && abiOrPref)
      llvm_unreachable("NYI");

    auto stTy = llvm::dyn_cast<mlir::cir::StructType>(ty);
    if (stTy && stTy.getPacked() && abiOrPref)
      return llvm::Align(1);

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *layout =
        getStructLayout(llvm::cast<mlir::cir::StructType>(ty));
    const llvm::Align align =
        abiOrPref ? structAlignment.ABIAlign : structAlignment.PrefAlign;
    return std::max(align, layout->getAlignment());
  }

  // FIXME(cir): This does not account for differnt address spaces, and relies
  // on CIR's data layout to give the proper alignment.
  assert(!::cir::MissingFeatures::addressSpace());

  // Fetch type alignment from MLIR's data layout.
  unsigned align = abiOrPref ? layout.getTypeABIAlignment(ty)
                             : layout.getTypePreferredAlignment(ty);
  return llvm::Align(align);
}

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
llvm::TypeSize CIRDataLayout::getTypeSizeInBits(mlir::Type ty) const {
  assert(!::cir::MissingFeatures::typeIsSized() &&
         "Cannot getTypeInfo() on a type that is unsized!");

  if (auto structTy = llvm::dyn_cast<mlir::cir::StructType>(ty)) {

    // FIXME(cir): CIR struct's data layout implementation doesn't do a good job
    // of handling unions particularities. We should have a separate union type.
    if (structTy.isUnion()) {
      auto largestMember = structTy.getLargestMember(layout);
      return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(largestMember));
    }

    // FIXME(cir): We should be able to query the size of a struct directly to
    // its data layout implementation instead of requiring a separate
    // StructLayout object.
    // Get the layout annotation... which is lazily created on demand.
    return getStructLayout(structTy)->getSizeInBits();
  }

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!::cir::MissingFeatures::addressSpace());

  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(ty));
}
