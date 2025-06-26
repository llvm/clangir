
#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "CIRGenTypes.h"

#include "TargetInfo.h"
#include "mlir/IR/BuiltinTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

using namespace llvm;
using namespace clang;
using namespace clang::CIRGen;

namespace {
/// The CIRRecordLowering is responsible for lowering an ASTRecordLayout to a
/// mlir::Type. Some of the lowering is straightforward, some is not. TODO: Here
/// we detail some of the complexities and weirdnesses?
struct CIRRecordLowering final {

  // MemberInfo is a helper structure that contains information about a record
  // member. In addition to the standard member types, there exists a sentinel
  // member type that ensures correct rounding.
  struct MemberInfo final {
    CharUnits offset;
    enum class InfoKind { VFPtr, VBPtr, Field, Base, VBase, Scissor } kind;
    mlir::Type data;
    union {
      const FieldDecl *fieldDecl;
      const CXXRecordDecl *cxxRecordDecl;
    };
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const FieldDecl *fieldDecl = nullptr)
        : offset{offset}, kind{kind}, data{data}, fieldDecl{fieldDecl} {};
    MemberInfo(CharUnits offset, InfoKind kind, mlir::Type data,
               const CXXRecordDecl *RD)
        : offset{offset}, kind{kind}, data{data}, cxxRecordDecl{RD} {}
    // MemberInfos are sorted so we define a < operator.
    bool operator<(const MemberInfo &other) const {
      return offset < other.offset;
    }
  };
  // The constructor.
  CIRRecordLowering(CIRGenTypes &cirGenTypes, const RecordDecl *recordDecl,
                    bool isPacked);

  /// ----------------------
  /// Short helper routines.

  /// Constructs a MemberInfo instance from an offset and mlir::Type.
  MemberInfo StorageInfo(CharUnits Offset, mlir::Type Data) {
    return MemberInfo(Offset, MemberInfo::InfoKind::Field, Data);
  }

  // Layout routines.
  void setBitFieldInfo(const FieldDecl *FD, CharUnits StartOffset,
                       mlir::Type StorageType);

  void lower(bool nonVirtualBaseType);
  void lowerUnion();

  /// Determines if we need a packed llvm struct.
  void determinePacked(bool NVBaseType);
  /// Inserts padding everywhere it's needed.
  void insertPadding();

  void computeVolatileBitfields();
  void accumulateBases();
  void accumulateVPtrs();
  void accumulateVBases();
  void accumulateFields();
  RecordDecl::field_iterator
  accumulateBitFields(RecordDecl::field_iterator Field,
                      RecordDecl::field_iterator FieldEnd);

  mlir::Type getVFPtrType();

  // Helper function to check if we are targeting AAPCS.
  bool isAAPCS() const {
    return astContext.getTargetInfo().getABI().starts_with("aapcs");
  }

  /// Helper function to check if the target machine is BigEndian.
  bool isBE() const { return astContext.getTargetInfo().isBigEndian(); }

  /// The Microsoft bitfield layout rule allocates discrete storage
  /// units of the field's formal type and only combines adjacent
  /// fields of the same formal type.  We want to emit a layout with
  /// these discrete storage units instead of combining them into a
  /// continuous run.
  bool isDiscreteBitFieldABI() {
    return astContext.getTargetInfo().getCXXABI().isMicrosoft() ||
           recordDecl->isMsStruct(astContext);
  }

  // The Itanium base layout rule allows virtual bases to overlap
  // other bases, which complicates layout in specific ways.
  //
  // Note specifically that the ms_struct attribute doesn't change this.
  bool isOverlappingVBaseABI() {
    return !astContext.getTargetInfo().getCXXABI().isMicrosoft();
  }
  // Recursively searches all of the bases to find out if a vbase is
  // not the primary vbase of some base class.
  bool hasOwnStorage(const CXXRecordDecl *Decl, const CXXRecordDecl *Query);

  CharUnits bitsToCharUnits(uint64_t bitOffset) {
    return astContext.toCharUnitsFromBits(bitOffset);
  }

  void calculateZeroInit();

  CharUnits getSize(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSize(Ty));
  }
  CharUnits getSizeInBits(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSizeInBits(Ty));
  }
  CharUnits getAlignment(mlir::Type Ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeABIAlignment(Ty));
  }
  bool isZeroInitializable(const FieldDecl *FD) {
    return cirGenTypes.isZeroInitializable(FD->getType());
  }
  bool isZeroInitializable(const RecordDecl *RD) {
    return cirGenTypes.isZeroInitializable(RD);
  }

  mlir::Type getCharType() {
    return cir::IntType::get(&cirGenTypes.getMLIRContext(),
                             astContext.getCharWidth(),
                             /*isSigned=*/false);
  }

  /// Wraps cir::IntType with some implicit arguments.
  mlir::Type getUIntNType(uint64_t NumBits) {
    unsigned AlignedBits = llvm::PowerOf2Ceil(NumBits);
    AlignedBits = std::max(8u, AlignedBits);
    return cir::IntType::get(&cirGenTypes.getMLIRContext(), AlignedBits,
                             /*isSigned=*/false);
  }

  mlir::Type getByteArrayType(CharUnits numberOfChars) {
    assert(!numberOfChars.isZero() && "Empty byte arrays aren't allowed.");
    mlir::Type type = getCharType();
    return numberOfChars == CharUnits::One()
               ? type
               : cir::ArrayType::get(type, numberOfChars.getQuantity());
  }

  // This is different from LLVM traditional codegen because CIRGen uses arrays
  // of bytes instead of arbitrary-sized integers. This is important for packed
  // structures support.
  mlir::Type getBitfieldStorageType(unsigned numBits) {
    unsigned alignedBits = llvm::alignTo(numBits, astContext.getCharWidth());
    if (cir::isValidFundamentalIntWidth(alignedBits)) {
      return builder.getUIntNTy(alignedBits);
    } else {
      mlir::Type type = getCharType();
      return cir::ArrayType::get(type, alignedBits / astContext.getCharWidth());
    }
  }

  // Gets the llvm Basesubobject type from a CXXRecordDecl.
  mlir::Type getStorageType(const CXXRecordDecl *RD) {
    return cirGenTypes.getCIRGenRecordLayout(RD).getBaseSubobjectCIRType();
  }

  mlir::Type getStorageType(const FieldDecl *fieldDecl) {
    auto type = cirGenTypes.convertTypeForMem(fieldDecl->getType());
    assert(!fieldDecl->isBitField() && "bit fields NYI");
    if (!fieldDecl->isBitField())
      return type;

    // if (isDiscreteBitFieldABI())
    //   return type;

    // return getUIntNType(std::min(fielddecl->getBitWidthValue(astContext),
    //     static_cast<unsigned int>(astContext.toBits(getSize(type)))));
    llvm_unreachable("getStorageType only supports nonBitFields at this point");
  }

  uint64_t getFieldBitOffset(const FieldDecl *fieldDecl) {
    return astRecordLayout.getFieldOffset(fieldDecl->getFieldIndex());
  }

  /// Fills out the structures that are ultimately consumed.
  void fillOutputFields();

  void appendPaddingBytes(CharUnits Size) {
    if (!Size.isZero()) {
      fieldTypes.push_back(getByteArrayType(Size));
      isPadded = 1;
    }
  }

  CIRGenTypes &cirGenTypes;
  CIRGenBuilderTy &builder;
  const ASTContext &astContext;
  const RecordDecl *recordDecl;
  const CXXRecordDecl *cxxRecordDecl;
  const ASTRecordLayout &astRecordLayout;
  // Helpful intermediate data-structures
  std::vector<MemberInfo> members;
  // Output fields, consumed by CIRGenTypes::computeRecordLayout
  llvm::SmallVector<mlir::Type, 16> fieldTypes;
  llvm::DenseMap<const FieldDecl *, unsigned> fields;
  llvm::DenseMap<const FieldDecl *, CIRGenBitFieldInfo> bitFields;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> nonVirtualBases;
  llvm::DenseMap<const CXXRecordDecl *, unsigned> virtualBases;
  cir::CIRDataLayout dataLayout;
  bool IsZeroInitializable : 1;
  bool IsZeroInitializableAsBase : 1;
  bool isPacked : 1;
  bool isPadded : 1;

private:
  CIRRecordLowering(const CIRRecordLowering &) = delete;
  void operator=(const CIRRecordLowering &) = delete;
};
} // namespace

CIRRecordLowering::CIRRecordLowering(CIRGenTypes &cirGenTypes,
                                     const RecordDecl *recordDecl,
                                     bool isPacked)
    : cirGenTypes{cirGenTypes}, builder{cirGenTypes.getBuilder()},
      astContext{cirGenTypes.getContext()}, recordDecl{recordDecl},
      cxxRecordDecl{llvm::dyn_cast<CXXRecordDecl>(recordDecl)},
      astRecordLayout{cirGenTypes.getContext().getASTRecordLayout(recordDecl)},
      dataLayout{cirGenTypes.getModule().getModule()},
      IsZeroInitializable(true), IsZeroInitializableAsBase(true),
      isPacked{isPacked}, isPadded{false} {}

void CIRRecordLowering::setBitFieldInfo(const FieldDecl *FD,
                                        CharUnits StartOffset,
                                        mlir::Type StorageType) {
  CIRGenBitFieldInfo &Info = bitFields[FD->getCanonicalDecl()];
  Info.IsSigned = FD->getType()->isSignedIntegerOrEnumerationType();
  Info.Offset =
      (unsigned)(getFieldBitOffset(FD) - astContext.toBits(StartOffset));
  Info.Size = FD->getBitWidthValue();
  Info.StorageSize = getSizeInBits(StorageType).getQuantity();
  Info.StorageOffset = StartOffset;
  Info.StorageType = StorageType;
  Info.Name = FD->getName();

  if (Info.Size > Info.StorageSize)
    Info.Size = Info.StorageSize;
  // Reverse the bit offsets for big endian machines. Because we represent
  // a bitfield as a single large integer load, we can imagine the bits
  // counting from the most-significant-bit instead of the
  // least-significant-bit.
  if (dataLayout.isBigEndian())
    Info.Offset = Info.StorageSize - (Info.Offset + Info.Size);

  Info.VolatileStorageSize = 0;
  Info.VolatileOffset = 0;
  Info.VolatileStorageOffset = CharUnits::Zero();
}

void CIRRecordLowering::lower(bool nonVirtualBaseType) {
  if (recordDecl->isUnion()) {
    lowerUnion();
    computeVolatileBitfields();
    return;
  }

  CharUnits Size = nonVirtualBaseType ? astRecordLayout.getNonVirtualSize()
                                      : astRecordLayout.getSize();
  accumulateFields();

  // RD implies C++
  if (cxxRecordDecl) {
    accumulateVPtrs();
    accumulateBases();
    if (members.empty()) {
      appendPaddingBytes(Size);
      computeVolatileBitfields();
      return;
    }
    if (!nonVirtualBaseType)
      accumulateVBases();
  }

  llvm::stable_sort(members);
  // TODO: implement clipTailPadding once bitfields are implemented
  // TODO: implemented packed records
  // TODO: implement padding
  // TODO: support zeroInit

  members.push_back(StorageInfo(Size, getUIntNType(8)));
  determinePacked(nonVirtualBaseType);
  insertPadding();
  members.pop_back();

  calculateZeroInit();
  fillOutputFields();
  computeVolatileBitfields();
}

void CIRRecordLowering::lowerUnion() {
  CharUnits LayoutSize = astRecordLayout.getSize();
  mlir::Type StorageType = nullptr;
  bool SeenNamedMember = false;
  // Iterate through the fields setting bitFieldInfo and the Fields array. Also
  // locate the "most appropriate" storage type.  The heuristic for finding the
  // storage type isn't necessary, the first (non-0-length-bitfield) field's
  // type would work fine and be simpler but would be different than what we've
  // been doing and cause lit tests to change.
  for (const auto *Field : recordDecl->fields()) {

    mlir::Type FieldType = nullptr;
    if (Field->isBitField()) {
      if (Field->isZeroLengthBitField())
        continue;

      FieldType = getBitfieldStorageType(Field->getBitWidthValue());

      setBitFieldInfo(Field, CharUnits::Zero(), FieldType);
    } else {
      FieldType = getStorageType(Field);
    }
    fields[Field->getCanonicalDecl()] = 0;
    // auto FieldType = getStorageType(Field);
    // Compute zero-initializable status.
    // This union might not be zero initialized: it may contain a pointer to
    // data member which might have some exotic initialization sequence.
    // If this is the case, then we aught not to try and come up with a "better"
    // type, it might not be very easy to come up with a Constant which
    // correctly initializes it.
    if (!SeenNamedMember) {
      SeenNamedMember = Field->getIdentifier();
      if (!SeenNamedMember)
        if (const auto *FieldRD = Field->getType()->getAsRecordDecl())
          SeenNamedMember = FieldRD->findFirstNamedDataMember();
      if (SeenNamedMember && !isZeroInitializable(Field)) {
        IsZeroInitializable = IsZeroInitializableAsBase = false;
        StorageType = FieldType;
      }
    }
    // Because our union isn't zero initializable, we won't be getting a better
    // storage type.
    if (!IsZeroInitializable)
      continue;

    // Conditionally update our storage type if we've got a new "better" one.
    if (!StorageType || getAlignment(FieldType) > getAlignment(StorageType) ||
        (getAlignment(FieldType) == getAlignment(StorageType) &&
         getSize(FieldType) > getSize(StorageType)))
      StorageType = FieldType;

    // NOTE(cir): Track all union member's types, not just the largest one. It
    // allows for proper type-checking and retain more info for analisys.
    fieldTypes.push_back(FieldType);
  }
  // If we have no storage type just pad to the appropriate size and return.
  if (!StorageType)
    return appendPaddingBytes(LayoutSize);
  // If our storage size was bigger than our required size (can happen in the
  // case of packed bitfields on Itanium) then just use an I8 array.
  if (LayoutSize < getSize(StorageType))
    StorageType = getByteArrayType(LayoutSize);
  // NOTE(cir): Defer padding calculations to the lowering process.
  appendPaddingBytes(LayoutSize - getSize(StorageType));
  // Set packed if we need it.
  if (LayoutSize % getAlignment(StorageType))
    isPacked = true;
}

bool CIRRecordLowering::hasOwnStorage(const CXXRecordDecl *Decl,
                                      const CXXRecordDecl *Query) {
  const ASTRecordLayout &DeclLayout = astContext.getASTRecordLayout(Decl);
  if (DeclLayout.isPrimaryBaseVirtual() && DeclLayout.getPrimaryBase() == Query)
    return false;
  for (const auto &Base : Decl->bases())
    if (!hasOwnStorage(Base.getType()->getAsCXXRecordDecl(), Query))
      return false;
  return true;
}

/// The AAPCS that defines that, when possible, bit-fields should
/// be accessed using containers of the declared type width:
/// When a volatile bit-field is read, and its container does not overlap with
/// any non-bit-field member or any zero length bit-field member, its container
/// must be read exactly once using the access width appropriate to the type of
/// the container. When a volatile bit-field is written, and its container does
/// not overlap with any non-bit-field member or any zero-length bit-field
/// member, its container must be read exactly once and written exactly once
/// using the access width appropriate to the type of the container. The two
/// accesses are not atomic.
///
/// Enforcing the width restriction can be disabled using
/// -fno-aapcs-bitfield-width.
void CIRRecordLowering::computeVolatileBitfields() {
  if (!isAAPCS() ||
      !cirGenTypes.getModule().getCodeGenOpts().AAPCSBitfieldWidth)
    return;

  for ([[maybe_unused]] auto &I : bitFields) {
    assert(!cir::MissingFeatures::armComputeVolatileBitfields());
  }
}

void CIRRecordLowering::accumulateBases() {
  // If we've got a primary virtual base, we need to add it with the bases.
  if (astRecordLayout.isPrimaryBaseVirtual()) {
    const CXXRecordDecl *BaseDecl = astRecordLayout.getPrimaryBase();
    members.push_back(MemberInfo(CharUnits::Zero(), MemberInfo::InfoKind::Base,
                                 getStorageType(BaseDecl), BaseDecl));
  }

  // Accumulate the non-virtual bases.
  for ([[maybe_unused]] const auto &Base : cxxRecordDecl->bases()) {
    if (Base.isVirtual())
      continue;
    // Bases can be zero-sized even if not technically empty if they
    // contain only a trailing array member.
    const CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl();
    if (!BaseDecl->isEmpty() &&
        !astContext.getASTRecordLayout(BaseDecl).getNonVirtualSize().isZero()) {
      members.push_back(MemberInfo(astRecordLayout.getBaseClassOffset(BaseDecl),
                                   MemberInfo::InfoKind::Base,
                                   getStorageType(BaseDecl), BaseDecl));
    }
  }
}

void CIRRecordLowering::accumulateVBases() {
  CharUnits ScissorOffset = astRecordLayout.getNonVirtualSize();
  // In the itanium ABI, it's possible to place a vbase at a dsize that is
  // smaller than the nvsize.  Here we check to see if such a base is placed
  // before the nvsize and set the scissor offset to that, instead of the
  // nvsize.
  if (isOverlappingVBaseABI())
    for (const auto &Base : cxxRecordDecl->vbases()) {
      const CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl();
      if (BaseDecl->isEmpty())
        continue;
      // If the vbase is a primary virtual base of some base, then it doesn't
      // get its own storage location but instead lives inside of that base.
      if (astContext.isNearlyEmpty(BaseDecl) &&
          !hasOwnStorage(cxxRecordDecl, BaseDecl))
        continue;
      ScissorOffset = std::min(ScissorOffset,
                               astRecordLayout.getVBaseClassOffset(BaseDecl));
    }
  members.push_back(MemberInfo(ScissorOffset, MemberInfo::InfoKind::Scissor,
                               mlir::Type{}, cxxRecordDecl));
  for (const auto &Base : cxxRecordDecl->vbases()) {
    const CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl();
    if (BaseDecl->isEmpty())
      continue;
    CharUnits Offset = astRecordLayout.getVBaseClassOffset(BaseDecl);
    // If the vbase is a primary virtual base of some base, then it doesn't
    // get its own storage location but instead lives inside of that base.
    if (isOverlappingVBaseABI() && astContext.isNearlyEmpty(BaseDecl) &&
        !hasOwnStorage(cxxRecordDecl, BaseDecl)) {
      members.push_back(
          MemberInfo(Offset, MemberInfo::InfoKind::VBase, nullptr, BaseDecl));
      continue;
    }
    // If we've got a vtordisp, add it as a storage type.
    if (astRecordLayout.getVBaseOffsetsMap()
            .find(BaseDecl)
            ->second.hasVtorDisp())
      members.push_back(
          StorageInfo(Offset - CharUnits::fromQuantity(4), getUIntNType(32)));
    members.push_back(MemberInfo(Offset, MemberInfo::InfoKind::VBase,
                                 getStorageType(BaseDecl), BaseDecl));
  }
}

void CIRRecordLowering::accumulateVPtrs() {
  if (astRecordLayout.hasOwnVFPtr())
    members.push_back(MemberInfo(CharUnits::Zero(), MemberInfo::InfoKind::VFPtr,
                                 getVFPtrType()));
  if (astRecordLayout.hasOwnVBPtr())
    llvm_unreachable("NYI");
}

mlir::Type CIRRecordLowering::getVFPtrType() {
  // FIXME: replay LLVM codegen for now, perhaps add a vtable ptr special
  // type so it's a bit more clear and C++ idiomatic.
  return builder.getVirtualFnPtrType();
}

void CIRRecordLowering::fillOutputFields() {
  for (auto &member : members) {
    if (member.data)
      fieldTypes.push_back(member.data);
    if (member.kind == MemberInfo::InfoKind::Field) {
      if (member.fieldDecl)
        fields[member.fieldDecl->getCanonicalDecl()] = fieldTypes.size() - 1;
      // A field without storage must be a bitfield.
      if (!member.data)
        setBitFieldInfo(member.fieldDecl, member.offset, fieldTypes.back());
    } else if (member.kind == MemberInfo::InfoKind::Base) {
      nonVirtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    } else if (member.kind == MemberInfo::InfoKind::VBase) {
      virtualBases[member.cxxRecordDecl] = fieldTypes.size() - 1;
    }
  }
}

RecordDecl::field_iterator
CIRRecordLowering::accumulateBitFields(RecordDecl::field_iterator Field,
                                       RecordDecl::field_iterator FieldEnd) {

  if (isDiscreteBitFieldABI())
    llvm_unreachable("NYI");

  CharUnits RegSize =
      bitsToCharUnits(astContext.getTargetInfo().getRegisterWidth());
  unsigned CharBits = astContext.getCharWidth();

  // Data about the start of the span we're accumulating to create an access
  // unit from. Begin is the first bitfield of the span. If Begin is FieldEnd,
  // we've not got a current span. The span starts at the BeginOffset character
  // boundary. BitSizeSinceBegin is the size (in bits) of the span -- this might
  // include padding when we've advanced to a subsequent bitfield run.
  RecordDecl::field_iterator Begin = FieldEnd;
  CharUnits BeginOffset;
  uint64_t BitSizeSinceBegin;

  // The (non-inclusive) end of the largest acceptable access unit we've found
  // since Begin. If this is Begin, we're gathering the initial set of bitfields
  // of a new span. BestEndOffset is the end of that acceptable access unit --
  // it might extend beyond the last character of the bitfield run, using
  // available padding characters.
  RecordDecl::field_iterator BestEnd = Begin;
  CharUnits BestEndOffset;
  bool BestClipped; // Whether the representation must be in a byte array.

  for (;;) {
    // AtAlignedBoundary is true if Field is the (potential) start of a new
    // span (or the end of the bitfields). When true, LimitOffset is the
    // character offset of that span and Barrier indicates whether the new
    // span cannot be merged into the current one.
    bool AtAlignedBoundary = false;
    bool Barrier = false; // a barrier can be a zero Bit Width or non bit member
    if (Field != FieldEnd && Field->isBitField()) {
      uint64_t BitOffset = getFieldBitOffset(*Field);
      if (Begin == FieldEnd) {
        // Beginning a new span.
        Begin = Field;
        BestEnd = Begin;

        assert((BitOffset % CharBits) == 0 && "Not at start of char");
        BeginOffset = bitsToCharUnits(BitOffset);
        BitSizeSinceBegin = 0;
      } else if ((BitOffset % CharBits) != 0) {
        // Bitfield occupies the same character as previous bitfield, it must be
        // part of the same span. This can include zero-length bitfields, should
        // the target not align them to character boundaries. Such non-alignment
        // is at variance with the standards, which require zero-length
        // bitfields be a barrier between access units. But of course we can't
        // achieve that in the middle of a character.
        assert(BitOffset ==
                   astContext.toBits(BeginOffset) + BitSizeSinceBegin &&
               "Concatenating non-contiguous bitfields");
      } else {
        // Bitfield potentially begins a new span. This includes zero-length
        // bitfields on non-aligning targets that lie at character boundaries
        // (those are barriers to merging).
        if (Field->isZeroLengthBitField())
          Barrier = true;
        AtAlignedBoundary = true;
      }
    } else {
      // We've reached the end of the bitfield run. Either we're done, or this
      // is a barrier for the current span.
      if (Begin == FieldEnd)
        break;

      Barrier = true;
      AtAlignedBoundary = true;
    }

    // InstallBest indicates whether we should create an access unit for the
    // current best span: fields [Begin, BestEnd) occupying characters
    // [BeginOffset, BestEndOffset).
    bool InstallBest = false;
    if (AtAlignedBoundary) {
      // Field is the start of a new span or the end of the bitfields. The
      // just-seen span now extends to BitSizeSinceBegin.

      // Determine if we can accumulate that just-seen span into the current
      // accumulation.
      CharUnits AccessSize = bitsToCharUnits(BitSizeSinceBegin + CharBits - 1);
      if (BestEnd == Begin) {
        // This is the initial run at the start of a new span. By definition,
        // this is the best seen so far.
        BestEnd = Field;
        BestEndOffset = BeginOffset + AccessSize;
        // Assume clipped until proven not below.
        BestClipped = true;
        if (!BitSizeSinceBegin)
          // A zero-sized initial span -- this will install nothing and reset
          // for another.
          InstallBest = true;
      } else if (AccessSize > RegSize) {
        // Accumulating the just-seen span would create a multi-register access
        // unit, which would increase register pressure.
        InstallBest = true;
      }

      if (!InstallBest) {
        // Determine if accumulating the just-seen span will create an expensive
        // access unit or not.
        mlir::Type Type = getUIntNType(astContext.toBits(AccessSize));
        if (!astContext.getTargetInfo().hasCheapUnalignedBitFieldAccess())
          llvm_unreachable("NYI");

        if (!InstallBest) {
          // Find the next used storage offset to determine what the limit of
          // the current span is. That's either the offset of the next field
          // with storage (which might be Field itself) or the end of the
          // non-reusable tail padding.
          CharUnits LimitOffset;
          for (auto Probe = Field; Probe != FieldEnd; ++Probe)
            if (!isEmptyFieldForLayout(astContext, *Probe)) {
              // A member with storage sets the limit.
              assert((getFieldBitOffset(*Probe) % CharBits) == 0 &&
                     "Next storage is not byte-aligned");
              LimitOffset = bitsToCharUnits(getFieldBitOffset(*Probe));
              goto FoundLimit;
            }
          LimitOffset = cxxRecordDecl ? astRecordLayout.getNonVirtualSize()
                                      : astRecordLayout.getDataSize();
        FoundLimit:
          CharUnits TypeSize = getSize(Type);
          if (BeginOffset + TypeSize <= LimitOffset) {
            // There is space before LimitOffset to create a naturally-sized
            // access unit.
            BestEndOffset = BeginOffset + TypeSize;
            BestEnd = Field;
            BestClipped = false;
          }
          if (Barrier) {
            // The next field is a barrier that we cannot merge across.
            InstallBest = true;
          } else if (cirGenTypes.getModule()
                         .getCodeGenOpts()
                         .FineGrainedBitfieldAccesses) {
            llvm_unreachable("NYI");
          } else {
            // Otherwise, we're not installing. Update the bit size
            // of the current span to go all the way to LimitOffset, which is
            // the (aligned) offset of next bitfield to consider.
            BitSizeSinceBegin = astContext.toBits(LimitOffset - BeginOffset);
          }
        }
      }
    }

    if (InstallBest) {
      assert((Field == FieldEnd || !Field->isBitField() ||
              (getFieldBitOffset(*Field) % CharBits) == 0) &&
             "Installing but not at an aligned bitfield or limit");
      CharUnits AccessSize = BestEndOffset - BeginOffset;
      if (!AccessSize.isZero()) {
        // Add the storage member for the access unit to the record. The
        // bitfields get the offset of their storage but come afterward and
        // remain there after a stable sort.
        mlir::Type Type;
        if (BestClipped) {
          assert(getSize(getUIntNType(astContext.toBits(AccessSize))) >
                     AccessSize &&
                 "Clipped access need not be clipped");
          Type = getByteArrayType(AccessSize);
        } else {
          Type = getUIntNType(astContext.toBits(AccessSize));
          assert(getSize(Type) == AccessSize &&
                 "Unclipped access must be clipped");
        }
        members.push_back(StorageInfo(BeginOffset, Type));
        for (; Begin != BestEnd; ++Begin)
          if (!Begin->isZeroLengthBitField())
            members.push_back(MemberInfo(
                BeginOffset, MemberInfo::InfoKind::Field, nullptr, *Begin));
      }
      // Reset to start a new span.
      Field = BestEnd;
      Begin = FieldEnd;
    } else {
      assert(Field != FieldEnd && Field->isBitField() &&
             "Accumulating past end of bitfields");
      assert(!Barrier && "Accumulating across barrier");
      // Accumulate this bitfield into the current (potential) span.
      BitSizeSinceBegin += Field->getBitWidthValue();
      ++Field;
    }
  }

  return Field;
}

void CIRRecordLowering::accumulateFields() {
  for (RecordDecl::field_iterator field = recordDecl->field_begin(),
                                  fieldEnd = recordDecl->field_end();
       field != fieldEnd;) {
    if (field->isBitField()) {
      field = accumulateBitFields(field, fieldEnd);
      assert((field == fieldEnd || !field->isBitField()) &&
             "Failed to accumulate all the bitfields");
    } else if (!field->isZeroSize(astContext)) {
      members.push_back(MemberInfo{bitsToCharUnits(getFieldBitOffset(*field)),
                                   MemberInfo::InfoKind::Field,
                                   getStorageType(*field), *field});
      ++field;
    } else {
      // TODO(cir): do we want to do anything special about zero size
      // members?
      ++field;
    }
  }
}

void CIRRecordLowering::calculateZeroInit() {
  for (const MemberInfo &member : members) {
    if (member.kind == MemberInfo::InfoKind::Field) {
      if (!member.fieldDecl || isZeroInitializable(member.fieldDecl))
        continue;
      IsZeroInitializable = IsZeroInitializableAsBase = false;
      return;
    } else if (member.kind == MemberInfo::InfoKind::Base ||
               member.kind == MemberInfo::InfoKind::VBase) {
      if (isZeroInitializable(member.cxxRecordDecl))
        continue;
      IsZeroInitializable = false;
      if (member.kind == MemberInfo::InfoKind::Base)
        IsZeroInitializableAsBase = false;
    }
  }
}

void CIRRecordLowering::determinePacked(bool NVBaseType) {
  if (isPacked)
    return;
  CharUnits Alignment = CharUnits::One();
  CharUnits NVAlignment = CharUnits::One();
  CharUnits NVSize = !NVBaseType && cxxRecordDecl
                         ? astRecordLayout.getNonVirtualSize()
                         : CharUnits::Zero();
  for (std::vector<MemberInfo>::const_iterator Member = members.begin(),
                                               MemberEnd = members.end();
       Member != MemberEnd; ++Member) {
    if (!Member->data)
      continue;
    // If any member falls at an offset that it not a multiple of its alignment,
    // then the entire record must be packed.
    if (Member->offset % getAlignment(Member->data))
      isPacked = true;
    if (Member->offset < NVSize)
      NVAlignment = std::max(NVAlignment, getAlignment(Member->data));
    Alignment = std::max(Alignment, getAlignment(Member->data));
  }
  // If the size of the record (the capstone's offset) is not a multiple of the
  // record's alignment, it must be packed.
  if (members.back().offset % Alignment)
    isPacked = true;
  // If the non-virtual sub-object is not a multiple of the non-virtual
  // sub-object's alignment, it must be packed.  We cannot have a packed
  // non-virtual sub-object and an unpacked complete object or vise versa.
  if (NVSize % NVAlignment)
    isPacked = true;
  // Update the alignment of the sentinel.
  if (!isPacked)
    members.back().data = getUIntNType(astContext.toBits(Alignment));
}

void CIRRecordLowering::insertPadding() {
  std::vector<std::pair<CharUnits, CharUnits>> Padding;
  CharUnits Size = CharUnits::Zero();
  for (std::vector<MemberInfo>::const_iterator Member = members.begin(),
                                               MemberEnd = members.end();
       Member != MemberEnd; ++Member) {
    if (!Member->data)
      continue;
    CharUnits Offset = Member->offset;
    assert(Offset >= Size);
    // Insert padding if we need to.
    if (Offset !=
        Size.alignTo(isPacked ? CharUnits::One() : getAlignment(Member->data)))
      Padding.push_back(std::make_pair(Size, Offset - Size));
    Size = Offset + getSize(Member->data);
  }
  if (Padding.empty())
    return;
  isPadded = 1;
  // Add the padding to the Members list and sort it.
  for (std::vector<std::pair<CharUnits, CharUnits>>::const_iterator
           Pad = Padding.begin(),
           PadEnd = Padding.end();
       Pad != PadEnd; ++Pad)
    members.push_back(StorageInfo(Pad->first, getByteArrayType(Pad->second)));
  llvm::stable_sort(members);
}

std::unique_ptr<CIRGenRecordLayout>
CIRGenTypes::computeRecordLayout(const RecordDecl *D, cir::RecordType *Ty) {
  CIRRecordLowering builder(*this, D, /*packed=*/false);
  assert(Ty->isIncomplete() && "recomputing record layout?");
  builder.lower(/*nonVirtualBaseType=*/false);

  // If we're in C++, compute the base subobject type.
  cir::RecordType BaseTy;
  if (llvm::isa<CXXRecordDecl>(D) && !D->isUnion() &&
      !D->hasAttr<FinalAttr>()) {
    BaseTy = *Ty;
    if (builder.astRecordLayout.getNonVirtualSize() !=
        builder.astRecordLayout.getSize()) {
      CIRRecordLowering baseBuilder(*this, D, /*Packed=*/builder.isPacked);
      baseBuilder.lower(/*NonVirtualBaseType=*/true);
      auto baseIdentifier = getRecordTypeName(D, ".base");
      BaseTy = Builder.getCompleteRecordTy(baseBuilder.fieldTypes,
                                           baseIdentifier, baseBuilder.isPacked,
                                           baseBuilder.isPadded, D);
      // TODO(cir): add something like addRecordTypeName

      // BaseTy and Ty must agree on their packedness for getCIRFieldNo to work
      // on both of them with the same index.
      assert(builder.isPacked == baseBuilder.isPacked &&
             "Non-virtual and complete types must agree on packedness");
    }
  }

  // Fill in the record *after* computing the base type.  Filling in the body
  // signifies that the type is no longer opaque and record layout is complete,
  // but we may need to recursively layout D while laying D out as a base type.
  auto astAttr = cir::ASTRecordDeclAttr::get(Ty->getContext(), D);
  Ty->complete(builder.fieldTypes, builder.isPacked, builder.isPadded, astAttr);

  auto RL = std::make_unique<CIRGenRecordLayout>(
      Ty ? *Ty : cir::RecordType{}, BaseTy ? BaseTy : cir::RecordType{},
      (bool)builder.IsZeroInitializable,
      (bool)builder.IsZeroInitializableAsBase);

  RL->NonVirtualBases.swap(builder.nonVirtualBases);
  RL->CompleteObjectVirtualBases.swap(builder.virtualBases);

  // Add all the field numbers.
  RL->FieldInfo.swap(builder.fields);

  // Add bitfield info.
  RL->BitFields.swap(builder.bitFields);

  // Dump the layout, if requested.
  if (getContext().getLangOpts().DumpRecordLayouts) {
    llvm::outs() << "\n*** Dumping CIRgen Record Layout\n";
    llvm::outs() << "Record: ";
    D->dump(llvm::outs());
    llvm::outs() << "\nLayout: ";
    RL->print(llvm::outs());
  }

  // TODO: implement verification
  return RL;
}

void CIRGenRecordLayout::print(raw_ostream &os) const {
  os << "<CIRecordLayout\n";
  os << "   CIR Type:" << CompleteObjectType << "\n";
  if (BaseSubobjectType)
    os << "   NonVirtualBaseCIRType:" << BaseSubobjectType << "\n";
  os << "   IsZeroInitializable:" << IsZeroInitializable << "\n";
  os << "   BitFields:[\n";
  std::vector<std::pair<unsigned, const CIRGenBitFieldInfo *>> bitInfo;
  for (auto &[decl, info] : BitFields) {
    const RecordDecl *rd = decl->getParent();
    unsigned index = 0;
    for (RecordDecl::field_iterator it = rd->field_begin(); *it != decl; ++it)
      ++index;
    bitInfo.push_back(std::make_pair(index, &info));
  }
  llvm::array_pod_sort(bitInfo.begin(), bitInfo.end());
  for (auto &info : bitInfo) {
    os.indent(4);
    info.second->print(os);
    os << "\n";
  }
  os << "   ]>\n";
}

void CIRGenRecordLayout::dump() const { print(llvm::errs()); }

void CIRGenBitFieldInfo::print(raw_ostream &os) const {
  os << "<CIRBitFieldInfo" << " name:" << Name << " offset:" << Offset
     << " size:" << Size << " isSigned:" << IsSigned
     << " storageSize:" << StorageSize
     << " storageOffset:" << StorageOffset.getQuantity()
     << " volatileOffset:" << VolatileOffset
     << " volatileStorageSize:" << VolatileStorageSize
     << " volatileStorageOffset:" << VolatileStorageOffset.getQuantity() << ">";
}

void CIRGenBitFieldInfo::dump() const { print(llvm::errs()); }

CIRGenBitFieldInfo CIRGenBitFieldInfo::MakeInfo(CIRGenTypes &Types,
                                                const FieldDecl *FD,
                                                uint64_t Offset, uint64_t Size,
                                                uint64_t StorageSize,
                                                CharUnits StorageOffset) {
  llvm_unreachable("NYI");
}
