#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/IR/DataLayout.h"

using namespace cir;

//===----------------------------------------------------------------------===//
// Support for RecordLayout
//===----------------------------------------------------------------------===//

RecordLayout::RecordLayout(cir::RecordType ST, const CIRDataLayout &DL)
    : RecordSize(llvm::TypeSize::getFixed(0)) {
  assert(!ST.isIncomplete() && "Cannot get layout of opaque records");
  IsPadded = false;
  NumElements = ST.getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    mlir::Type Ty = ST.getMembers()[i];
    if (i == 0 && cir::MissingFeatures::typeIsScalableType())
      llvm_unreachable("Scalable types are not yet supported in CIR");

    assert(!cir::MissingFeatures::recordDeclIsPacked() &&
           "Cannot identify packed records");
    const llvm::Align TyAlign =
        ST.getPacked() ? llvm::Align(1) : DL.getABITypeAlign(Ty);

    // Add padding if necessary to align the data element properly.
    // Currently the only record with scalable size will be the homogeneous
    // scalable vector types. Homogeneous scalable vector types have members of
    // the same data type so no alignment issue will happen. The condition here
    // assumes so and needs to be adjusted if this assumption changes (e.g. we
    // support records with arbitrary scalable data type, or record that
    // contains both fixed size and scalable size data type members).
    if (!RecordSize.isScalable() && !isAligned(TyAlign, RecordSize)) {
      IsPadded = true;
      RecordSize = llvm::TypeSize::getFixed(alignTo(RecordSize, TyAlign));
    }

    // Keep track of maximum alignment constraint.
    RecordAlignment = std::max(TyAlign, RecordAlignment);

    getMemberOffsets()[i] = RecordSize;
    // Consume space for this data item
    RecordSize += DL.getTypeAllocSize(Ty);
  }

  // Add padding to the end of the record so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!RecordSize.isScalable() && !isAligned(RecordAlignment, RecordSize)) {
    IsPadded = true;
    RecordSize = llvm::TypeSize::getFixed(alignTo(RecordSize, RecordAlignment));
  }
}

/// getElementContainingOffset - Given a valid offset into the record,
/// return the record index that contains it.
unsigned RecordLayout::getElementContainingOffset(uint64_t FixedOffset) const {
  assert(!RecordSize.isScalable() &&
         "Cannot get element at offset for record containing scalable "
         "vector types");
  llvm::TypeSize Offset = llvm::TypeSize::getFixed(FixedOffset);
  llvm::ArrayRef<llvm::TypeSize> MemberOffsets = getMemberOffsets();

  const auto *SI =
      std::upper_bound(MemberOffsets.begin(), MemberOffsets.end(), Offset,
                       [](llvm::TypeSize LHS, llvm::TypeSize RHS) -> bool {
                         return llvm::TypeSize::isKnownLT(LHS, RHS);
                       });
  assert(SI != MemberOffsets.begin() && "Offset not in record type!");
  --SI;
  assert(llvm::TypeSize::isKnownLE(*SI, Offset) && "upper_bound didn't work");
  assert((SI == MemberOffsets.begin() ||
          llvm::TypeSize::isKnownLE(*(SI - 1), Offset)) &&
         (SI + 1 == MemberOffsets.end() ||
          llvm::TypeSize::isKnownGT(*(SI + 1), Offset)) &&
         "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI - MemberOffsets.begin();
}

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

namespace {

class RecordLayoutMap {
  using LayoutInfoTy = llvm::DenseMap<cir::RecordType, RecordLayout *>;
  LayoutInfoTy LayoutInfo;

public:
  ~RecordLayoutMap() {
    // Remove any layouts.
    for (const auto &I : LayoutInfo) {
      RecordLayout *Value = I.second;
      Value->~RecordLayout();
      free(Value);
    }
  }

  RecordLayout *&operator[](cir::RecordType STy) { return LayoutInfo[STy]; }
};

} // namespace

CIRDataLayout::CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} {
  reset(modOp.getDataLayoutSpec());
  if (auto attr = modOp->getAttr(cir::CIRDialect::getTypeSizeInfoAttrName()))
    typeSizeInfo = mlir::cast<TypeSizeInfoAttr>(attr);
  else {
    // Generate default size information.
    auto voidPtrTy = PointerType::get(VoidType::get(modOp->getContext()));
    llvm::TypeSize ptrSize = getTypeSizeInBits(voidPtrTy);
    typeSizeInfo =
        TypeSizeInfoAttr::get(modOp->getContext(),
                              /*char_size=*/8, /*int_size=*/32,
                              /*size_t_size=*/ptrSize.getFixedValue());
  }
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

  LayoutMap = nullptr;

  // ManglingMode = MM_None;
  // NonIntegralAddressSpaces.clear();
  RecordAlignment =
      llvm::DataLayout::PrimitiveSpec{0, llvm::Align(1), llvm::Align(8)};

  // NOTE(cir): Alignment setter functions are skipped as these should already
  // be set in MLIR's data layout.
}

void CIRDataLayout::clear() {
  delete static_cast<RecordLayoutMap *>(LayoutMap);
  LayoutMap = nullptr;
}

const RecordLayout *CIRDataLayout::getRecordLayout(cir::RecordType Ty) const {
  if (!LayoutMap)
    LayoutMap = new RecordLayoutMap();

  RecordLayoutMap *STM = static_cast<RecordLayoutMap *>(LayoutMap);
  RecordLayout *&SL = (*STM)[Ty];
  if (SL)
    return SL;

  // Otherwise, create the record layout.  Because it is variable length, we
  // malloc it, then use placement new.
  RecordLayout *L = (RecordLayout *)llvm::safe_malloc(
      RecordLayout::totalSizeToAlloc<llvm::TypeSize>(Ty.getNumElements()));

  // Set SL before calling RecordLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) RecordLayout(Ty, *this);

  return L;
}

/*!
  \param abiOrPref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abiOrPref == true) or preferred alignment (\a abiOrPref
  == false) for the requested type \a Ty.
 */
llvm::Align CIRDataLayout::getAlignment(mlir::Type Ty, bool abiOrPref) const {

  if (llvm::isa<cir::RecordType>(Ty)) {
    // Packed record types always have an ABI alignment of one.
    if (cir::MissingFeatures::recordDeclIsPacked() && abiOrPref)
      llvm_unreachable("NYI");

    auto stTy = llvm::dyn_cast<cir::RecordType>(Ty);
    if (stTy && stTy.getPacked() && abiOrPref)
      return llvm::Align(1);

    // Get the layout annotation... which is lazily created on demand.
    const RecordLayout *Layout =
        getRecordLayout(llvm::cast<cir::RecordType>(Ty));
    const llvm::Align Align =
        abiOrPref ? RecordAlignment.ABIAlign : RecordAlignment.PrefAlign;
    return std::max(Align, Layout->getAlignment());
  }

  // FIXME(cir): This does not account for differnt address spaces, and relies
  // on CIR's data layout to give the proper alignment.
  assert(!cir::MissingFeatures::addressSpace());

  // Fetch type alignment from MLIR's data layout.
  unsigned align = abiOrPref ? layout.getTypeABIAlignment(Ty)
                             : layout.getTypePreferredAlignment(Ty);
  return llvm::Align(align);
}

// The implementation of this method is provided inline as it is particularly
// well suited to constant folding when called on a specific Type subclass.
llvm::TypeSize CIRDataLayout::getTypeSizeInBits(mlir::Type Ty) const {
  assert(!cir::MissingFeatures::typeIsSized() &&
         "Cannot getTypeInfo() on a type that is unsized!");

  if (auto recordTy = llvm::dyn_cast<cir::RecordType>(Ty)) {
    // FIXME(cir): CIR record's data layout implementation doesn't do a good job
    // of handling unions particularities. We should have a separate union type.
    return recordTy.getTypeSizeInBits(layout, {});
  }

  // FIXME(cir): This does not account for different address spaces, and relies
  // on CIR's data layout to give the proper ABI-specific type width.
  assert(!cir::MissingFeatures::addressSpace());

  return llvm::TypeSize::getFixed(layout.getTypeSizeInBits(Ty));
}
