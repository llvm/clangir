
#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "CIRGenTypes.h"

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
using namespace cir;

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
               const CXXRecordDecl *rd)
        : offset{offset}, kind{kind}, data{data}, cxxRecordDecl{rd} {}
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
  MemberInfo storageInfo(CharUnits offset, mlir::Type data) {
    return MemberInfo(offset, MemberInfo::InfoKind::Field, data);
  }

  // Layout routines.
  void setBitFieldInfo(const FieldDecl *fd, CharUnits startOffset,
                       mlir::Type storageType);

  void lower(bool nonVirtualBaseType);
  void lowerUnion();

  /// Determines if we need a packed llvm struct.
  void determinePacked(bool nvBaseType);
  /// Inserts padding everywhere it's needed.
  void insertPadding();

  void computeVolatileBitfields();
  void accumulateBases();
  void accumulateVPtrs();
  void accumulateVBases();
  void accumulateFields();
  void accumulateBitFields(RecordDecl::field_iterator field,
                           RecordDecl::field_iterator fieldEnd);

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
  bool hasOwnStorage(const CXXRecordDecl *decl, const CXXRecordDecl *query);

  CharUnits bitsToCharUnits(uint64_t bitOffset) {
    return astContext.toCharUnitsFromBits(bitOffset);
  }

  void calculateZeroInit();

  CharUnits getSize(mlir::Type ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSize(ty));
  }
  CharUnits getSizeInBits(mlir::Type ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeSizeInBits(ty));
  }
  CharUnits getAlignment(mlir::Type ty) {
    return CharUnits::fromQuantity(dataLayout.layout.getTypeABIAlignment(ty));
  }
  bool isZeroInitializable(const FieldDecl *fd) {
    return cirGenTypes.isZeroInitializable(fd->getType());
  }
  bool isZeroInitializable(const RecordDecl *rd) {
    return cirGenTypes.isZeroInitializable(rd);
  }

  mlir::Type getCharType() {
    return mlir::cir::IntType::get(&cirGenTypes.getMLIRContext(),
                                   astContext.getCharWidth(),
                                   /*isSigned=*/false);
  }

  /// Wraps mlir::cir::IntType with some implicit arguments.
  mlir::Type getUIntNType(uint64_t numBits) {
    unsigned alignedBits = llvm::PowerOf2Ceil(numBits);
    alignedBits = std::max(8u, alignedBits);
    return mlir::cir::IntType::get(&cirGenTypes.getMLIRContext(), alignedBits,
                                   /*isSigned=*/false);
  }

  mlir::Type getByteArrayType(CharUnits numberOfChars) {
    assert(!numberOfChars.isZero() && "Empty byte arrays aren't allowed.");
    mlir::Type type = getCharType();
    return numberOfChars == CharUnits::One()
               ? type
               : mlir::cir::ArrayType::get(type.getContext(), type,
                                           numberOfChars.getQuantity());
  }

  // This is different from LLVM traditional codegen because CIRGen uses arrays
  // of bytes instead of arbitrary-sized integers. This is important for packed
  // structures support.
  mlir::Type getBitfieldStorageType(unsigned numBits) {
    unsigned alignedBits = llvm::alignTo(numBits, astContext.getCharWidth());
    if (mlir::cir::IntType::isValidPrimitiveIntBitwidth(alignedBits)) {
      return builder.getUIntNTy(alignedBits);
    }
    mlir::Type type = getCharType();
    return mlir::cir::ArrayType::get(type.getContext(), type,
                                     alignedBits / astContext.getCharWidth());
  }

  // Gets the llvm Basesubobject type from a CXXRecordDecl.
  mlir::Type getStorageType(const CXXRecordDecl *rd) {
    return cirGenTypes.getCIRGenRecordLayout(rd).getBaseSubobjectCIRType();
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

  void appendPaddingBytes(CharUnits size) {
    if (!size.isZero())
      fieldTypes.push_back(getByteArrayType(size));
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
  CIRDataLayout dataLayout;
  bool isZeroInitable : 1;
  bool isZeroInitializableAsBase : 1;
  bool isPacked : 1;

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
      dataLayout{cirGenTypes.getModule().getModule()}, isZeroInitable(true),
      isZeroInitializableAsBase(true), isPacked{isPacked} {}

void CIRRecordLowering::setBitFieldInfo(const FieldDecl *fd,
                                        CharUnits startOffset,
                                        mlir::Type storageType) {
  CIRGenBitFieldInfo &info = bitFields[fd->getCanonicalDecl()];
  info.IsSigned = fd->getType()->isSignedIntegerOrEnumerationType();
  info.Offset =
      (unsigned)(getFieldBitOffset(fd) - astContext.toBits(startOffset));
  info.Size = fd->getBitWidthValue(astContext);
  info.StorageSize = getSizeInBits(storageType).getQuantity();
  info.StorageOffset = startOffset;
  info.StorageType = storageType;
  info.Name = fd->getName();

  if (info.Size > info.StorageSize)
    info.Size = info.StorageSize;
  // Reverse the bit offsets for big endian machines. Because we represent
  // a bitfield as a single large integer load, we can imagine the bits
  // counting from the most-significant-bit instead of the
  // least-significant-bit.
  if (dataLayout.isBigEndian())
    info.Offset = info.StorageSize - (info.Offset + info.Size);

  info.VolatileStorageSize = 0;
  info.VolatileOffset = 0;
  info.VolatileStorageOffset = CharUnits::Zero();
}

void CIRRecordLowering::lower(bool nonVirtualBaseType) {
  if (recordDecl->isUnion()) {
    lowerUnion();
    computeVolatileBitfields();
    return;
  }

  CharUnits size = nonVirtualBaseType ? astRecordLayout.getNonVirtualSize()
                                      : astRecordLayout.getSize();
  accumulateFields();

  // RD implies C++
  if (cxxRecordDecl) {
    accumulateVPtrs();
    accumulateBases();
    if (members.empty()) {
      appendPaddingBytes(size);
      computeVolatileBitfields();
      return;
    }
    if (!nonVirtualBaseType)
      accumulateVBases();
  }

  llvm::stable_sort(members);
  // TODO: implement clipTailPadding once bitfields are implemented
  // TODO: implemented packed structs
  // TODO: implement padding
  // TODO: support zeroInit

  members.push_back(storageInfo(size, getUIntNType(8)));
  determinePacked(nonVirtualBaseType);
  insertPadding();
  members.pop_back();

  fillOutputFields();
  computeVolatileBitfields();
}

void CIRRecordLowering::lowerUnion() {
  CharUnits layoutSize = astRecordLayout.getSize();
  mlir::Type storageType = nullptr;
  bool seenNamedMember = false;
  // Iterate through the fields setting bitFieldInfo and the Fields array. Also
  // locate the "most appropriate" storage type.  The heuristic for finding the
  // storage type isn't necessary, the first (non-0-length-bitfield) field's
  // type would work fine and be simpler but would be different than what we've
  // been doing and cause lit tests to change.
  for (const auto *field : recordDecl->fields()) {

    mlir::Type fieldType = nullptr;
    if (field->isBitField()) {
      if (field->isZeroLengthBitField(astContext))
        continue;

      fieldType = getBitfieldStorageType(field->getBitWidthValue(astContext));

      setBitFieldInfo(field, CharUnits::Zero(), fieldType);
    } else {
      fieldType = getStorageType(field);
    }
    fields[field->getCanonicalDecl()] = 0;
    // auto FieldType = getStorageType(Field);
    // Compute zero-initializable status.
    // This union might not be zero initialized: it may contain a pointer to
    // data member which might have some exotic initialization sequence.
    // If this is the case, then we aught not to try and come up with a "better"
    // type, it might not be very easy to come up with a Constant which
    // correctly initializes it.
    if (!seenNamedMember) {
      seenNamedMember = field->getIdentifier();
      if (!seenNamedMember)
        if (const auto *fieldRd = field->getType()->getAsRecordDecl())
          seenNamedMember = fieldRd->findFirstNamedDataMember();
      if (seenNamedMember && !isZeroInitializable(field)) {
        isZeroInitable = isZeroInitializableAsBase = false;
        storageType = fieldType;
      }
    }
    // Because our union isn't zero initializable, we won't be getting a better
    // storage type.
    if (!isZeroInitable)
      continue;

    // Conditionally update our storage type if we've got a new "better" one.
    if (!storageType || getAlignment(fieldType) > getAlignment(storageType) ||
        (getAlignment(fieldType) == getAlignment(storageType) &&
         getSize(fieldType) > getSize(storageType)))
      storageType = fieldType;

    // NOTE(cir): Track all union member's types, not just the largest one. It
    // allows for proper type-checking and retain more info for analisys.
    fieldTypes.push_back(fieldType);
  }
  // If we have no storage type just pad to the appropriate size and return.
  if (!storageType)
    llvm_unreachable("no-storage union NYI");
  // If our storage size was bigger than our required size (can happen in the
  // case of packed bitfields on Itanium) then just use an I8 array.
  if (layoutSize < getSize(storageType))
    storageType = getByteArrayType(layoutSize);
  // NOTE(cir): Defer padding calculations to the lowering process.
  // appendPaddingBytes(LayoutSize - getSize(StorageType));
  // Set packed if we need it.
  if (layoutSize % getAlignment(storageType))
    isPacked = true;
}

bool CIRRecordLowering::hasOwnStorage(const CXXRecordDecl *decl,
                                      const CXXRecordDecl *query) {
  const ASTRecordLayout &declLayout = astContext.getASTRecordLayout(decl);
  if (declLayout.isPrimaryBaseVirtual() && declLayout.getPrimaryBase() == query)
    return false;
  for (const auto &base : decl->bases())
    if (!hasOwnStorage(base.getType()->getAsCXXRecordDecl(), query))
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

  for ([[maybe_unused]] auto &i : bitFields) {
    assert(!MissingFeatures::armComputeVolatileBitfields());
  }
}

void CIRRecordLowering::accumulateBases() {
  // If we've got a primary virtual base, we need to add it with the bases.
  if (astRecordLayout.isPrimaryBaseVirtual()) {
    llvm_unreachable("NYI");
  }

  // Accumulate the non-virtual bases.
  for ([[maybe_unused]] const auto &base : cxxRecordDecl->bases()) {
    if (base.isVirtual())
      continue;
    // Bases can be zero-sized even if not technically empty if they
    // contain only a trailing array member.
    const CXXRecordDecl *baseDecl = base.getType()->getAsCXXRecordDecl();
    if (!baseDecl->isEmpty() &&
        !astContext.getASTRecordLayout(baseDecl).getNonVirtualSize().isZero()) {
      members.emplace_back(astRecordLayout.getBaseClassOffset(baseDecl),
                           MemberInfo::InfoKind::Base, getStorageType(baseDecl),
                           baseDecl);
    }
  }
}

void CIRRecordLowering::accumulateVBases() {
  CharUnits scissorOffset = astRecordLayout.getNonVirtualSize();
  // In the itanium ABI, it's possible to place a vbase at a dsize that is
  // smaller than the nvsize.  Here we check to see if such a base is placed
  // before the nvsize and set the scissor offset to that, instead of the
  // nvsize.
  if (isOverlappingVBaseABI())
    for (const auto &base : cxxRecordDecl->vbases()) {
      const CXXRecordDecl *baseDecl = base.getType()->getAsCXXRecordDecl();
      if (baseDecl->isEmpty())
        continue;
      // If the vbase is a primary virtual base of some base, then it doesn't
      // get its own storage location but instead lives inside of that base.
      if (astContext.isNearlyEmpty(baseDecl) &&
          !hasOwnStorage(cxxRecordDecl, baseDecl))
        continue;
      scissorOffset = std::min(scissorOffset,
                               astRecordLayout.getVBaseClassOffset(baseDecl));
    }
  members.emplace_back(scissorOffset, MemberInfo::InfoKind::Scissor,
                       mlir::Type{}, cxxRecordDecl);
  for (const auto &base : cxxRecordDecl->vbases()) {
    const CXXRecordDecl *baseDecl = base.getType()->getAsCXXRecordDecl();
    if (baseDecl->isEmpty())
      continue;
    CharUnits offset = astRecordLayout.getVBaseClassOffset(baseDecl);
    // If the vbase is a primary virtual base of some base, then it doesn't
    // get its own storage location but instead lives inside of that base.
    if (isOverlappingVBaseABI() && astContext.isNearlyEmpty(baseDecl) &&
        !hasOwnStorage(cxxRecordDecl, baseDecl)) {
      members.emplace_back(offset, MemberInfo::InfoKind::VBase, nullptr,
                           baseDecl);
      continue;
    }
    // If we've got a vtordisp, add it as a storage type.
    if (astRecordLayout.getVBaseOffsetsMap()
            .find(baseDecl)
            ->second.hasVtorDisp())
      members.push_back(
          storageInfo(offset - CharUnits::fromQuantity(4), getUIntNType(32)));
    members.emplace_back(offset, MemberInfo::InfoKind::VBase,
                         getStorageType(baseDecl), baseDecl);
  }
}

void CIRRecordLowering::accumulateVPtrs() {
  if (astRecordLayout.hasOwnVFPtr())
    members.emplace_back(CharUnits::Zero(), MemberInfo::InfoKind::VFPtr,
                         getVFPtrType());
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

void CIRRecordLowering::accumulateBitFields(
    RecordDecl::field_iterator field, RecordDecl::field_iterator fieldEnd) {
  // Run stores the first element of the current run of bitfields.  FieldEnd is
  // used as a special value to note that we don't have a current run.  A
  // bitfield run is a contiguous collection of bitfields that can be stored in
  // the same storage block.  Zero-sized bitfields and bitfields that would
  // cross an alignment boundary break a run and start a new one.
  RecordDecl::field_iterator run = fieldEnd;
  // Tail is the offset of the first bit off the end of the current run.  It's
  // used to determine if the ASTRecordLayout is treating these two bitfields as
  // contiguous.  StartBitOffset is offset of the beginning of the Run.
  uint64_t startBitOffset, tail = 0;
  if (isDiscreteBitFieldABI()) {
    llvm_unreachable("NYI");
  }

  // Check if OffsetInRecord (the size in bits of the current run) is better
  // as a single field run. When OffsetInRecord has legal integer width, and
  // its bitfield offset is naturally aligned, it is better to make the
  // bitfield a separate storage component so as it can be accessed directly
  // with lower cost.
  auto isBetterAsSingleFieldRun = [&](uint64_t offsetInRecord,
                                      uint64_t startBitOffset,
                                      uint64_t nextTail = 0) {
    if (!cirGenTypes.getModule().getCodeGenOpts().FineGrainedBitfieldAccesses)
      return false;
    llvm_unreachable("NYI");
    // if (OffsetInRecord < 8 || !llvm::isPowerOf2_64(OffsetInRecord) ||
    //     !DataLayout.fitsInLegalInteger(OffsetInRecord))
    //   return false;
    // Make sure StartBitOffset is naturally aligned if it is treated as an
    // IType integer.
    // if (StartBitOffset %
    //         astContext.toBits(getAlignment(getUIntNType(OffsetInRecord))) !=
    //     0)
    //   return false;
    return true;
  };

  // The start field is better as a single field run.
  bool startFieldAsSingleRun = false;
  for (;;) {
    // Check to see if we need to start a new run.
    if (run == fieldEnd) {
      // If we're out of fields, return.
      if (field == fieldEnd)
        break;
      // Any non-zero-length bitfield can start a new run.
      if (!field->isZeroLengthBitField(astContext)) {
        run = field;
        startBitOffset = getFieldBitOffset(*field);
        tail = startBitOffset + field->getBitWidthValue(astContext);
        startFieldAsSingleRun =
            isBetterAsSingleFieldRun(tail - startBitOffset, startBitOffset);
      }
      ++field;
      continue;
    }

    // If the start field of a new run is better as a single run, or if current
    // field (or consecutive fields) is better as a single run, or if current
    // field has zero width bitfield and either UseZeroLengthBitfieldAlignment
    // or UseBitFieldTypeAlignment is set to true, or if the offset of current
    // field is inconsistent with the offset of previous field plus its offset,
    // skip the block below and go ahead to emit the storage. Otherwise, try to
    // add bitfields to the run.
    uint64_t nextTail = tail;
    if (field != fieldEnd)
      nextTail += field->getBitWidthValue(astContext);

    if (!startFieldAsSingleRun && field != fieldEnd &&
        !isBetterAsSingleFieldRun(tail - startBitOffset, startBitOffset,
                                  nextTail) &&
        (!field->isZeroLengthBitField(astContext) ||
         (!astContext.getTargetInfo().useZeroLengthBitfieldAlignment() &&
          !astContext.getTargetInfo().useBitFieldTypeAlignment())) &&
        tail == getFieldBitOffset(*field)) {
      tail = nextTail;
      ++field;
      continue;
    }

    // We've hit a break-point in the run and need to emit a storage field.
    auto type = getBitfieldStorageType(tail - startBitOffset);

    // Add the storage member to the record and set the bitfield info for all of
    // the bitfields in the run. Bitfields get the offset of their storage but
    // come afterward and remain there after a stable sort.
    members.push_back(storageInfo(bitsToCharUnits(startBitOffset), type));
    for (; run != field; ++run)
      members.emplace_back(bitsToCharUnits(startBitOffset),
                           MemberInfo::InfoKind::Field, nullptr, *run);
    run = fieldEnd;
    startFieldAsSingleRun = false;
  }
}

void CIRRecordLowering::accumulateFields() {
  for (RecordDecl::field_iterator field = recordDecl->field_begin(),
                                  fieldEnd = recordDecl->field_end();
       field != fieldEnd;) {
    if (field->isBitField()) {
      RecordDecl::field_iterator start = field;
      // Iterate to gather the list of bitfields.
      for (++field; field != fieldEnd && field->isBitField(); ++field)
        ;
      accumulateBitFields(start, field);
    } else if (!field->isZeroSize(astContext)) {
      members.emplace_back(bitsToCharUnits(getFieldBitOffset(*field)),
                           MemberInfo::InfoKind::Field, getStorageType(*field),
                           *field);
      ++field;
    } else {
      // TODO(cir): do we want to do anything special about zero size
      // members?
      ++field;
    }
  }
}

void CIRRecordLowering::determinePacked(bool nvBaseType) {
  if (isPacked)
    return;
  CharUnits alignment = CharUnits::One();
  CharUnits nvAlignment = CharUnits::One();
  CharUnits nvSize = !nvBaseType && cxxRecordDecl
                         ? astRecordLayout.getNonVirtualSize()
                         : CharUnits::Zero();
  for (const auto &member : members) {
    if (!member.data)
      continue;
    // If any member falls at an offset that it not a multiple of its alignment,
    // then the entire record must be packed.
    if (member.offset % getAlignment(member.data))
      isPacked = true;
    if (member.offset < nvSize)
      nvAlignment = std::max(nvAlignment, getAlignment(member.data));
    alignment = std::max(alignment, getAlignment(member.data));
  }
  // If the size of the record (the capstone's offset) is not a multiple of the
  // record's alignment, it must be packed.
  if (members.back().offset % alignment)
    isPacked = true;
  // If the non-virtual sub-object is not a multiple of the non-virtual
  // sub-object's alignment, it must be packed.  We cannot have a packed
  // non-virtual sub-object and an unpacked complete object or vise versa.
  if (nvSize % nvAlignment)
    isPacked = true;
  // Update the alignment of the sentinel.
  if (!isPacked)
    members.back().data = getUIntNType(astContext.toBits(alignment));
}

void CIRRecordLowering::insertPadding() {
  std::vector<std::pair<CharUnits, CharUnits>> padding;
  CharUnits size = CharUnits::Zero();
  for (const auto &member : members) {
    if (!member.data)
      continue;
    CharUnits offset = member.offset;
    assert(offset >= size);
    // Insert padding if we need to.
    if (offset !=
        size.alignTo(isPacked ? CharUnits::One() : getAlignment(member.data)))
      padding.emplace_back(size, offset - size);
    size = offset + getSize(member.data);
  }
  if (padding.empty())
    return;
  // Add the padding to the Members list and sort it.
  for (const auto &pad : padding)
    members.push_back(storageInfo(pad.first, getByteArrayType(pad.second)));
  llvm::stable_sort(members);
}

std::unique_ptr<CIRGenRecordLayout>
CIRGenTypes::computeRecordLayout(const RecordDecl *d,
                                 mlir::cir::StructType *ty) {
  CIRRecordLowering builder(*this, d, /*isPacked=*/false);
  assert(ty->isIncomplete() && "recomputing record layout?");
  builder.lower(/*nonVirtualBaseType=*/false);

  // If we're in C++, compute the base subobject type.
  mlir::cir::StructType baseTy;
  if (llvm::isa<CXXRecordDecl>(d) && !d->isUnion() &&
      !d->hasAttr<FinalAttr>()) {
    baseTy = *ty;
    if (builder.astRecordLayout.getNonVirtualSize() !=
        builder.astRecordLayout.getSize()) {
      CIRRecordLowering baseBuilder(*this, d, /*isPacked=*/builder.isPacked);
      baseBuilder.lower(/*NonVirtualBaseType=*/true);
      auto baseIdentifier = getRecordTypeName(d, ".base");
      baseTy = Builder.getCompleteStructTy(
          baseBuilder.fieldTypes, baseIdentifier, baseBuilder.isPacked, d);
      // TODO(cir): add something like addRecordTypeName

      // BaseTy and Ty must agree on their packedness for getCIRFieldNo to work
      // on both of them with the same index.
      assert(builder.isPacked == baseBuilder.isPacked &&
             "Non-virtual and complete types must agree on packedness");
    }
  }

  // Fill in the struct *after* computing the base type.  Filling in the body
  // signifies that the type is no longer opaque and record layout is complete,
  // but we may need to recursively layout D while laying D out as a base type.
  auto astAttr = mlir::cir::ASTRecordDeclAttr::get(ty->getContext(), d);
  ty->complete(builder.fieldTypes, builder.isPacked, astAttr);

  auto rl = std::make_unique<CIRGenRecordLayout>(
      ty ? *ty : mlir::cir::StructType{},
      baseTy ? baseTy : mlir::cir::StructType{}, (bool)builder.isZeroInitable,
      (bool)builder.isZeroInitializableAsBase);

  rl->NonVirtualBases.swap(builder.nonVirtualBases);
  rl->CompleteObjectVirtualBases.swap(builder.virtualBases);

  // Add all the field numbers.
  rl->FieldInfo.swap(builder.fields);

  // Add bitfield info.
  rl->BitFields.swap(builder.bitFields);

  // Dump the layout, if requested.
  if (getContext().getLangOpts().DumpRecordLayouts) {
    llvm_unreachable("NYI");
  }

  // TODO: implement verification
  return rl;
}

CIRGenBitFieldInfo CIRGenBitFieldInfo::MakeInfo(CIRGenTypes &types,
                                                const FieldDecl *fd,
                                                uint64_t offset, uint64_t size,
                                                uint64_t storageSize,
                                                CharUnits storageOffset) {
  llvm_unreachable("NYI");
}
