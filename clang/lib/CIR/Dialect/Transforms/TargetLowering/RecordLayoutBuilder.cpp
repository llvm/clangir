//=== RecordLayoutBuilder.cpp - Helper class for building record layouts ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/CGRecordLayoutBuilder.cpp. The
// queries are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRLowerContext.h"
#include "CIRRecordLayout.h"
#include "mlir/IR/Types.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

using namespace mlir;
using namespace mlir::cir;

namespace {

//===-----------------------------------------------------------------------==//
// EmptySubobjectMap Implementation
//===----------------------------------------------------------------------===//

/// Keeps track of which empty subobjects exist at different offsets while
/// laying out a C++ class.
class EmptySubobjectMap {
  [[maybe_unused]] const CIRLowerContext &context;
  uint64_t charWidth;

  /// The class whose empty entries we're keeping track of.
  const StructType klass;

  /// The highest offset known to contain an empty base subobject.
  clang::CharUnits maxEmptyClassOffset;

  /// Compute the size of the largest base or member subobject that is empty.
  void computeEmptySubobjectSizes();

public:
  /// This holds the size of the largest empty subobject (either a base
  /// or a member). Will be zero if the record being built doesn't contain
  /// any empty classes.
  clang::CharUnits sizeOfLargestEmptySubobject;

  EmptySubobjectMap(const CIRLowerContext &context, const StructType klass)
      : context(context), charWidth(context.getCharWidth()), klass(klass) {
    computeEmptySubobjectSizes();
  }

  /// Return whether a field can be placed at the given offset.
  bool canPlaceFieldAtOffset(Type ty, clang::CharUnits offset);
};

void EmptySubobjectMap::computeEmptySubobjectSizes() {
  // Check the bases.
  assert(!::cir::MissingFeatures::getCXXRecordBases());

  // Check the fields.
  for (const auto ft : klass.getMembers()) {
    assert(!::cir::MissingFeatures::qualifiedTypes());
    const auto rt = dyn_cast<StructType>(ft);

    // We only care about record types.
    if (!rt)
      continue;

    // TODO(cir): Handle nested record types.
    llvm_unreachable("NYI");
  }
}

bool EmptySubobjectMap::canPlaceFieldAtOffset(const Type ty,
                                              clang::CharUnits offset) {
  llvm_unreachable("NYI");
}

//===-----------------------------------------------------------------------==//
// ItaniumRecordLayoutBuilder Implementation
//===----------------------------------------------------------------------===//

class ItaniumRecordLayoutBuilder {
protected:
  // FIXME(cir):  Remove this and make the appropriate fields public.
  friend class mlir::cir::CIRLowerContext;

  const CIRLowerContext &context;

  EmptySubobjectMap *emptySubobjects;

  /// Size - The current size of the record layout.
  uint64_t size = 0;

  /// Alignment - The current alignment of the record layout.
  clang::CharUnits alignment;

  /// PreferredAlignment - The preferred alignment of the record layout.
  clang::CharUnits preferredAlignment;

  /// The alignment if attribute packed is not used.
  clang::CharUnits unpackedAlignment;

  /// \brief The maximum of the alignments of top-level members.
  clang::CharUnits unadjustedAlignment;

  SmallVector<uint64_t, 16> fieldOffsets;

  /// Whether the external AST source has provided a layout for this
  /// record.
  unsigned useExternalLayout : 1;

  /// Whether we need to infer alignment, even when we have an
  /// externally-provided layout.
  unsigned inferAlignment : 1;

  /// Packed - Whether the record is packed or not.
  unsigned packed : 1;

  unsigned isUnion : 1;

  unsigned isMac68kAlign : 1;

  unsigned isNaturalAlign : 1;

  unsigned isMsStruct : 1;

  /// UnfilledBitsInLastUnit - If the last field laid out was a bitfield,
  /// this contains the number of bits in the last unit that can be used for
  /// an adjacent bitfield if necessary.  The unit in question is usually
  /// a byte, but larger units are used if IsMsStruct.
  unsigned char unfilledBitsInLastUnit = 0;

  /// LastBitfieldStorageUnitSize - If IsMsStruct, represents the size of the
  /// storage unit of the previous field if it was a bitfield.
  unsigned char lastBitfieldStorageUnitSize = 0;

  /// MaxFieldAlignment - The maximum allowed field alignment. This is set by
  /// #pragma pack.
  clang::CharUnits maxFieldAlignment;

  /// DataSize - The data size of the record being laid out.
  uint64_t dataSize = 0;

  clang::CharUnits nonVirtualSize;
  clang::CharUnits nonVirtualAlignment;
  clang::CharUnits preferredNvAlignment;

  /// If we've laid out a field but not included its tail padding in Size yet,
  /// this is the size up to the end of that field.
  clang::CharUnits paddedFieldSize;

  /// The primary base class (if one exists) of the class we're laying out.
  const StructType primaryBase;

  /// Whether the primary base of the class we're laying out is virtual.
  bool primaryBaseIsVirtual = false;

  /// Whether the class provides its own vtable/vftbl pointer, as opposed to
  /// inheriting one from a primary base class.
  bool hasOwnVfPtr = false;

  /// the flag of field offset changing due to packed attribute.
  bool hasPackedField = false;

  /// An auxiliary field used for AIX. When there are OverlappingEmptyFields
  /// existing in the aggregate, the flag shows if the following first non-empty
  /// or empty-but-non-overlapping field has been handled, if any.
  bool handledFirstNonOverlappingEmptyField = false;

public:
  ItaniumRecordLayoutBuilder(const CIRLowerContext &context,
                             EmptySubobjectMap *emptySubobjects)
      : context(context), emptySubobjects(emptySubobjects),
        alignment(clang::CharUnits::One()),
        preferredAlignment(clang::CharUnits::One()),
        unpackedAlignment(clang::CharUnits::One()),
        unadjustedAlignment(clang::CharUnits::One()), useExternalLayout(false),
        inferAlignment(false), packed(false), isUnion(false),
        isMac68kAlign(false),
        isNaturalAlign(!context.getTargetInfo().getTriple().isOSAIX()),
        isMsStruct(false), maxFieldAlignment(clang::CharUnits::Zero()),
        nonVirtualSize(clang::CharUnits::Zero()),
        nonVirtualAlignment(clang::CharUnits::One()),
        preferredNvAlignment(clang::CharUnits::One()),
        paddedFieldSize(clang::CharUnits::Zero()) {}

  void layout(StructType rt);

  void layoutFields(StructType d);
  void layoutField(Type d, bool insertExtraPadding);

  void updateAlignment(clang::CharUnits newAlignment,
                       clang::CharUnits unpackedNewAlignment,
                       clang::CharUnits preferredNewAlignment);

  void checkFieldPadding(uint64_t offset, uint64_t unpaddedOffset,
                         uint64_t unpackedOffset, unsigned unpackedAlign,
                         bool isPacked, Type ty);

  clang::CharUnits getSize() const {
    assert(size % context.getCharWidth() == 0);
    return context.toCharUnitsFromBits(size);
  }
  uint64_t getSizeInBits() const { return size; }

  void setSize(clang::CharUnits newSize) { size = context.toBits(newSize); }
  void setSize(uint64_t newSize) { size = newSize; }

  clang::CharUnits getDataSize() const {
    assert(dataSize % context.getCharWidth() == 0);
    return context.toCharUnitsFromBits(dataSize);
  }

  /// Initialize record layout for the given record decl.
  void initializeLayout(Type ty);

  uint64_t getDataSizeInBits() const { return dataSize; }

  void setDataSize(clang::CharUnits newSize) {
    dataSize = context.toBits(newSize);
  }
  void setDataSize(uint64_t newSize) { dataSize = newSize; }
};

void ItaniumRecordLayoutBuilder::layout(const StructType rt) {
  initializeLayout(rt);

  // Lay out the vtable and the non-virtual bases.
  assert(!::cir::MissingFeatures::isCXXRecordDecl() &&
         !::cir::MissingFeatures::cxxRecordIsDynamicClass());

  layoutFields(rt);

  // FIXME(cir): Handle virtual-related layouts.
  assert(!::cir::MissingFeatures::getCXXRecordBases());

  assert(!::cir::MissingFeatures::itaniumRecordLayoutBuilderFinishLayout());
}

void ItaniumRecordLayoutBuilder::initializeLayout(const mlir::Type ty) {
  if (const auto rt = dyn_cast<StructType>(ty)) {
    isUnion = rt.isUnion();
    assert(!::cir::MissingFeatures::recordDeclIsMSStruct());
  }

  assert(!::cir::MissingFeatures::recordDeclIsPacked());

  // Honor the default struct packing maximum alignment flag.
  if (unsigned defaultMaxFieldAlignment = context.getLangOpts().PackStruct) {
    llvm_unreachable("NYI");
  }

  // mac68k alignment supersedes maximum field alignment and attribute aligned,
  // and forces all structures to have 2-byte alignment. The IBM docs on it
  // allude to additional (more complicated) semantics, especially with regard
  // to bit-fields, but gcc appears not to follow that.
  if (::cir::MissingFeatures::declHasAlignMac68kAttr()) {
    llvm_unreachable("NYI");
  } else {
    if (::cir::MissingFeatures::declHasAlignNaturalAttr())
      llvm_unreachable("NYI");

    if (::cir::MissingFeatures::declHasMaxFieldAlignmentAttr())
      llvm_unreachable("NYI");

    if (::cir::MissingFeatures::declGetMaxAlignment())
      llvm_unreachable("NYI");
  }

  handledFirstNonOverlappingEmptyField =
      !context.getTargetInfo().defaultsToAIXPowerAlignment() || isNaturalAlign;

  // If there is an external AST source, ask it for the various offsets.
  if (const auto rt = dyn_cast<StructType>(ty)) {
    if (::cir::MissingFeatures::astContextGetExternalSource()) {
      llvm_unreachable("NYI");
    }
  }
}

void ItaniumRecordLayoutBuilder::layoutField(const Type d,
                                             bool insertExtraPadding) {
  // auto FieldClass = D.dyn_cast<StructType>();
  assert(!::cir::MissingFeatures::fieldDeclIsPotentiallyOverlapping() &&
         !::cir::MissingFeatures::cxxRecordDeclIsEmptyCxX11());
  bool isOverlappingEmptyField = false; // FIXME(cir): Needs more features.

  clang::CharUnits fieldOffset = (isUnion || isOverlappingEmptyField)
                                     ? clang::CharUnits::Zero()
                                     : getDataSize();

  const bool defaultsToAixPowerAlignment =
      context.getTargetInfo().defaultsToAIXPowerAlignment();
  bool foundFirstNonOverlappingEmptyFieldForAix = false;
  if (defaultsToAixPowerAlignment && !handledFirstNonOverlappingEmptyField) {
    llvm_unreachable("NYI");
  }

  assert(!::cir::MissingFeatures::fieldDeclIsBitfield());

  uint64_t unpaddedFieldOffset = getDataSizeInBits() - unfilledBitsInLastUnit;
  // Reset the unfilled bits.
  unfilledBitsInLastUnit = 0;
  lastBitfieldStorageUnitSize = 0;

  llvm::Triple target = context.getTargetInfo().getTriple();

  clang::AlignRequirementKind alignRequirement =
      clang::AlignRequirementKind::None;
  clang::CharUnits fieldSize;
  clang::CharUnits fieldAlign;
  // The amount of this class's dsize occupied by the field.
  // This is equal to FieldSize unless we're permitted to pack
  // into the field's tail padding.
  clang::CharUnits effectiveFieldSize;

  auto setDeclInfo = [&](bool isIncompleteArrayType) {
    auto ti = context.getTypeInfoInChars(d);
    fieldAlign = ti.Align;
    // Flexible array members don't have any size, but they have to be
    // aligned appropriately for their element type.
    effectiveFieldSize = fieldSize =
        isIncompleteArrayType ? clang::CharUnits::Zero() : ti.Width;
    alignRequirement = ti.AlignRequirement;
  };

  if (isa<ArrayType>(d) && cast<ArrayType>(d).getSize() == 0) {
    llvm_unreachable("NYI");
  } else {
    setDeclInfo(false /* IsIncompleteArrayType */);

    if (::cir::MissingFeatures::fieldDeclIsPotentiallyOverlapping())
      llvm_unreachable("NYI");

    if (isMsStruct)
      llvm_unreachable("NYI");
  }

  assert(!::cir::MissingFeatures::recordDeclIsPacked() &&
         !::cir::MissingFeatures::cxxRecordDeclIsPod());
  bool fieldPacked = false; // FIXME(cir): Needs more features.

  // When used as part of a typedef, or together with a 'packed' attribute, the
  // 'aligned' attribute can be used to decrease alignment. In that case, it
  // overrides any computed alignment we have, and there is no need to upgrade
  // the alignment.
  auto alignedAttrCanDecreaseAIXAlignment = [alignRequirement, fieldPacked] {
    // Enum alignment sources can be safely ignored here, because this only
    // helps decide whether we need the AIX alignment upgrade, which only
    // applies to floating-point types.
    return alignRequirement == clang::AlignRequirementKind::RequiredByTypedef ||
           (alignRequirement == clang::AlignRequirementKind::RequiredByRecord &&
            fieldPacked);
  };

  // The AIX `power` alignment rules apply the natural alignment of the
  // "first member" if it is of a floating-point data type (or is an aggregate
  // whose recursively "first" member or element is such a type). The alignment
  // associated with these types for subsequent members use an alignment value
  // where the floating-point data type is considered to have 4-byte alignment.
  //
  // For the purposes of the foregoing: vtable pointers, non-empty base classes,
  // and zero-width bit-fields count as prior members; members of empty class
  // types marked `no_unique_address` are not considered to be prior members.
  clang::CharUnits preferredAlign = fieldAlign;
  if (defaultsToAixPowerAlignment && !alignedAttrCanDecreaseAIXAlignment() &&
      (foundFirstNonOverlappingEmptyFieldForAix || isNaturalAlign)) {
    llvm_unreachable("NYI");
  }

  // The align if the field is not packed. This is to check if the attribute
  // was unnecessary (-Wpacked).
  clang::CharUnits unpackedFieldAlign = fieldAlign;
  clang::CharUnits packedFieldAlign = clang::CharUnits::One();
  clang::CharUnits unpackedFieldOffset = fieldOffset;
  // clang::CharUnits OriginalFieldAlign = UnpackedFieldAlign;

  assert(!::cir::MissingFeatures::fieldDeclGetMaxFieldAlignment());
  clang::CharUnits maxAlignmentInChars = clang::CharUnits::Zero();
  packedFieldAlign = std::max(packedFieldAlign, maxAlignmentInChars);
  preferredAlign = std::max(preferredAlign, maxAlignmentInChars);
  unpackedFieldAlign = std::max(unpackedFieldAlign, maxAlignmentInChars);

  // The maximum field alignment overrides the aligned attribute.
  if (!maxFieldAlignment.isZero()) {
    llvm_unreachable("NYI");
  }

  if (!fieldPacked)
    fieldAlign = unpackedFieldAlign;
  if (defaultsToAixPowerAlignment)
    llvm_unreachable("NYI");
  if (fieldPacked) {
    llvm_unreachable("NYI");
  }

  clang::CharUnits alignTo =
      !defaultsToAixPowerAlignment ? fieldAlign : preferredAlign;
  // Round up the current record size to the field's alignment boundary.
  fieldOffset = fieldOffset.alignTo(alignTo);
  unpackedFieldOffset = unpackedFieldOffset.alignTo(unpackedFieldAlign);

  if (useExternalLayout) {
    llvm_unreachable("NYI");
  } else {
    if (!isUnion && emptySubobjects) {
      // Check if we can place the field at this offset.
      while (/*!EmptySubobjects->CanPlaceFieldAtOffset(D, FieldOffset)*/
             false) {
        llvm_unreachable("NYI");
      }
    }
  }

  // Place this field at the current location.
  fieldOffsets.push_back(context.toBits(fieldOffset));

  if (!useExternalLayout)
    checkFieldPadding(context.toBits(fieldOffset), unpaddedFieldOffset,
                      context.toBits(unpackedFieldOffset),
                      context.toBits(unpackedFieldAlign), fieldPacked, d);

  if (insertExtraPadding) {
    llvm_unreachable("NYI");
  }

  // Reserve space for this field.
  if (!isOverlappingEmptyField) {
    // uint64_t EffectiveFieldSizeInBits = Context.toBits(EffectiveFieldSize);
    if (isUnion)
      llvm_unreachable("NYI");
    else
      setDataSize(fieldOffset + effectiveFieldSize);

    paddedFieldSize = std::max(paddedFieldSize, fieldOffset + fieldSize);
    setSize(std::max(getSizeInBits(), getDataSizeInBits()));
  } else {
    llvm_unreachable("NYI");
  }

  // Remember max struct/class ABI-specified alignment.
  unadjustedAlignment = std::max(unadjustedAlignment, fieldAlign);
  updateAlignment(fieldAlign, unpackedFieldAlign, preferredAlign);

  // For checking the alignment of inner fields against
  // the alignment of its parent record.
  // FIXME(cir): We need to track the parent record of the current type being
  // laid out. A regular mlir::Type has not way of doing this. In fact, we will
  // likely need an external abstraction, as I don't think this is possible with
  // just the field type.
  assert(!::cir::MissingFeatures::fieldDeclAbstraction());

  if (packed && !fieldPacked && packedFieldAlign < fieldAlign)
    llvm_unreachable("NYI");
}

void ItaniumRecordLayoutBuilder::layoutFields(const StructType d) {
  // Layout each field, for now, just sequentially, respecting alignment.  In
  // the future, this will need to be tweakable by targets.
  assert(!::cir::MissingFeatures::recordDeclMayInsertExtraPadding() &&
         !context.getLangOpts().SanitizeAddressFieldPadding);
  bool insertExtraPadding = false;
  assert(!::cir::MissingFeatures::recordDeclHasFlexibleArrayMember());
  bool hasFlexibleArrayMember = false;
  for (const auto ft : d.getMembers()) {
    layoutField(ft, insertExtraPadding && (ft != d.getMembers().back() ||
                                           !hasFlexibleArrayMember));
  }
}

void ItaniumRecordLayoutBuilder::updateAlignment(
    clang::CharUnits newAlignment, clang::CharUnits unpackedNewAlignment,
    clang::CharUnits preferredNewAlignment) {
  // The alignment is not modified when using 'mac68k' alignment or when
  // we have an externally-supplied layout that also provides overall alignment.
  if (isMac68kAlign || (useExternalLayout && !inferAlignment))
    return;

  if (newAlignment > alignment) {
    assert(llvm::isPowerOf2_64(newAlignment.getQuantity()) &&
           "Alignment not a power of 2");
    alignment = newAlignment;
  }

  if (unpackedNewAlignment > unpackedAlignment) {
    assert(llvm::isPowerOf2_64(unpackedNewAlignment.getQuantity()) &&
           "Alignment not a power of 2");
    unpackedAlignment = unpackedNewAlignment;
  }

  if (preferredNewAlignment > preferredAlignment) {
    assert(llvm::isPowerOf2_64(preferredNewAlignment.getQuantity()) &&
           "Alignment not a power of 2");
    preferredAlignment = preferredNewAlignment;
  }
}

void ItaniumRecordLayoutBuilder::checkFieldPadding(
    uint64_t offset, uint64_t unpaddedOffset, uint64_t unpackedOffset,
    unsigned unpackedAlign, bool isPacked, const Type ty) {
  // We let objc ivars without warning, objc interfaces generally are not used
  // for padding tricks.
  if (::cir::MissingFeatures::objCIvarDecls())
    llvm_unreachable("NYI");

  // FIXME(cir): Should the following be skiped in CIR?
  // Don't warn about structs created without a SourceLocation.  This can
  // be done by clients of the AST, such as codegen.

  unsigned charBitNum = context.getTargetInfo().getCharWidth();

  // Warn if padding was introduced to the struct/class.
  if (!isUnion && offset > unpaddedOffset) {
    unsigned padSize = offset - unpaddedOffset;
    // bool InBits = true;
    if (padSize % charBitNum == 0) {
      padSize = padSize / charBitNum;
      // InBits = false;
    }
    assert(::cir::MissingFeatures::bitFieldPaddingDiagnostics());
  }
  if (isPacked && offset != unpackedOffset) {
    hasPackedField = true;
  }
}

//===-----------------------------------------------------------------------==//
// Misc. Helper Functions
//===----------------------------------------------------------------------===//

bool isMsLayout(const CIRLowerContext &context) {
  return context.getTargetInfo().getCXXABI().isMicrosoft();
}

/// Does the target C++ ABI require us to skip over the tail-padding
/// of the given class (considering it as a base class) when allocating
/// objects?
static bool mustSkipTailPadding(clang::TargetCXXABI abi, const StructType rd) {
  assert(!::cir::MissingFeatures::recordDeclIsCXXDecl());
  switch (abi.getTailPaddingUseRules()) {
  case clang::TargetCXXABI::AlwaysUseTailPadding:
    return false;

  case clang::TargetCXXABI::UseTailPaddingUnlessPOD03:
    // http://itanium-cxx-abi.github.io/cxx-abi/abi.html#POD :
    //   In general, a type is considered a POD for the purposes of
    //   layout if it is a POD type (in the sense of ISO C++
    //   [basic.types]). However, a POD-struct or POD-union (in the
    //   sense of ISO C++ [class]) with a bitfield member whose
    //   declared width is wider than the declared type of the
    //   bitfield is not a POD for the purpose of layout.  Similarly,
    //   an array type is not a POD for the purpose of layout if the
    //   element type of the array is not a POD for the purpose of
    //   layout.
    //
    //   Where references to the ISO C++ are made in this paragraph,
    //   the Technical Corrigendum 1 version of the standard is
    //   intended.
    // FIXME(cir): This always returns true since we can't check if a CIR record
    // is a POD type.
    assert(!::cir::MissingFeatures::cxxRecordDeclIsPod());
    return true;

  case clang::TargetCXXABI::UseTailPaddingUnlessPOD11:
    // This is equivalent to RD->getTypeForDecl().isCXX11PODType(),
    // but with a lot of abstraction penalty stripped off.  This does
    // assume that these properties are set correctly even in C++98
    // mode; fortunately, that is true because we want to assign
    // consistently semantics to the type-traits intrinsics (or at
    // least as many of them as possible).
    llvm_unreachable("NYI");
  }

  llvm_unreachable("bad tail-padding use kind");
}

} // namespace

/// Get or compute information about the layout of the specified record
/// (struct/union/class), which indicates its size and field position
/// information.
const CIRRecordLayout &CIRLowerContext::getCIRRecordLayout(const Type d) const {
  assert(isa<StructType>(d) && "Not a record type");
  auto rt = dyn_cast<StructType>(d);

  assert(rt.isComplete() && "Cannot get layout of forward declarations!");

  // FIXME(cir): Use a more MLIR-based approach by using it's buitin data layout
  // features, such as interfaces, cacheing, and the DLTI dialect.

  const CIRRecordLayout *newEntry = nullptr;

  if (isMsLayout(*this)) {
    llvm_unreachable("NYI");
  } else {
    // FIXME(cir): Add if-else separating C and C++ records.
    assert(!::cir::MissingFeatures::isCXXRecordDecl());
    EmptySubobjectMap emptySubobjects(*this, rt);
    ItaniumRecordLayoutBuilder builder(*this, &emptySubobjects);
    builder.layout(rt);

    // In certain situations, we are allowed to lay out objects in the
    // tail-padding of base classes.  This is ABI-dependent.
    // FIXME: this should be stored in the record layout.
    bool skipTailPadding = mustSkipTailPadding(getTargetInfo().getCXXABI(), rt);

    // FIXME: This should be done in FinalizeLayout.
    clang::CharUnits dataSize =
        skipTailPadding ? builder.getSize() : builder.getDataSize();
    clang::CharUnits nonVirtualSize =
        skipTailPadding ? dataSize : builder.nonVirtualSize;
    assert(!::cir::MissingFeatures::cxxRecordIsDynamicClass());
    // FIXME(cir): Whose responsible for freeing the allocation below?
    newEntry = new CIRRecordLayout(
        *this, builder.getSize(), builder.alignment, builder.preferredAlignment,
        builder.unadjustedAlignment,
        /*RequiredAlignment : used by MS-ABI)*/
        builder.alignment, builder.hasOwnVfPtr, /*RD->isDynamicClass()=*/false,
        clang::CharUnits::fromQuantity(-1), dataSize, builder.fieldOffsets,
        nonVirtualSize, builder.nonVirtualAlignment,
        builder.preferredNvAlignment,
        emptySubobjects.sizeOfLargestEmptySubobject, builder.primaryBase,
        builder.primaryBaseIsVirtual, nullptr, false, false);
  }

  // TODO(cir): Add option to dump the layouts.
  assert(!::cir::MissingFeatures::cacheRecordLayouts());

  return *newEntry;
}
