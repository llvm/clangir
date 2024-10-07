//===---- CIRGenExprCst.cpp - Emit LLVM Code from Constant Expressions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Constant Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//
#include "Address.h"
#include "CIRGenCXXABI.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Specifiers.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

using namespace clang;
using namespace cir;

//===----------------------------------------------------------------------===//
//                            ConstantAggregateBuilder
//===----------------------------------------------------------------------===//

namespace {
class ConstExprEmitter;

static mlir::Attribute
buildArrayConstant(CIRGenModule &cgm, mlir::Type desiredType,
                   mlir::Type commonElementType, unsigned arrayBound,
                   SmallVectorImpl<mlir::TypedAttr> &elements,
                   mlir::TypedAttr filler);

struct ConstantAggregateBuilderUtils {
  CIRGenModule &cgm;
  CIRDataLayout dataLayout;

  ConstantAggregateBuilderUtils(CIRGenModule &cgm)
      : cgm(cgm), dataLayout{cgm.getModule()} {}

  CharUnits getAlignment(const mlir::TypedAttr c) const {
    return CharUnits::fromQuantity(
        dataLayout.getAlignment(c.getType(), /*useABI=*/true));
  }

  CharUnits getSize(mlir::Type ty) const {
    return CharUnits::fromQuantity(dataLayout.getTypeAllocSize(ty));
  }

  CharUnits getSize(const mlir::TypedAttr c) const {
    return getSize(c.getType());
  }

  mlir::TypedAttr getPadding(CharUnits size) const {
    auto eltTy = cgm.UCharTy;
    auto arSize = size.getQuantity();
    auto &bld = cgm.getBuilder();
    SmallVector<mlir::Attribute, 4> elts(arSize, bld.getZeroAttr(eltTy));
    return bld.getConstArray(mlir::ArrayAttr::get(bld.getContext(), elts),
                             bld.getArrayType(eltTy, arSize));
  }

  mlir::Attribute getZeroes(CharUnits zeroSize) const {
    llvm_unreachable("NYI");
  }
};

/// Incremental builder for an mlir::TypedAttr holding a struct or array
/// constant.
class ConstantAggregateBuilder : private ConstantAggregateBuilderUtils {
  /// The elements of the constant. These two arrays must have the same size;
  /// Offsets[i] describes the offset of Elems[i] within the constant. The
  /// elements are kept in increasing offset order, and we ensure that there
  /// is no overlap: Offsets[i+1] >= Offsets[i] + getSize(Elemes[i]).
  ///
  /// This may contain explicit padding elements (in order to create a
  /// natural layout), but need not. Gaps between elements are implicitly
  /// considered to be filled with undef.
  llvm::SmallVector<mlir::Attribute, 32> elems;
  llvm::SmallVector<CharUnits, 32> offsets;

  /// The size of the constant (the maximum end offset of any added element).
  /// May be larger than the end of Elems.back() if we split the last element
  /// and removed some trailing undefs.
  CharUnits size = CharUnits::Zero();

  /// This is true only if laying out Elems in order as the elements of a
  /// non-packed LLVM struct will give the correct layout.
  bool naturalLayout = true;

  bool split(size_t index, CharUnits hint);
  std::optional<size_t> splitAt(CharUnits pos);

  static mlir::Attribute
  buildFrom(CIRGenModule &cgm, ArrayRef<mlir::Attribute> elems,
            ArrayRef<CharUnits> offsets, CharUnits startOffset, CharUnits size,
            bool naturalLayout, mlir::Type desiredTy, bool allowOversized);

public:
  ConstantAggregateBuilder(CIRGenModule &cgm)
      : ConstantAggregateBuilderUtils(cgm) {}

  /// Update or overwrite the value starting at \p Offset with \c C.
  ///
  /// \param AllowOverwrite If \c true, this constant might overwrite (part of)
  ///        a constant that has already been added. This flag is only used to
  ///        detect bugs.
  bool add(mlir::Attribute c, CharUnits offset, bool allowOverwrite);

  /// Update or overwrite the bits starting at \p OffsetInBits with \p Bits.
  bool addBits(llvm::APInt bits, uint64_t offsetInBits, bool allowOverwrite);

  /// Attempt to condense the value starting at \p Offset to a constant of type
  /// \p DesiredTy.
  void condense(CharUnits offset, mlir::Type desiredTy);

  /// Produce a constant representing the entire accumulated value, ideally of
  /// the specified type. If \p AllowOversized, the constant might be larger
  /// than implied by \p DesiredTy (eg, if there is a flexible array member).
  /// Otherwise, the constant will be of exactly the same size as \p DesiredTy
  /// even if we can't represent it as that type.
  mlir::Attribute build(mlir::Type desiredTy, bool allowOversized) const {
    return buildFrom(cgm, elems, offsets, CharUnits::Zero(), size,
                     naturalLayout, desiredTy, allowOversized);
  }
};

template <typename Container, typename Range = std::initializer_list<
                                  typename Container::value_type>>
static void replace(Container &c, size_t beginOff, size_t endOff, Range vals) {
  assert(beginOff <= endOff && "invalid replacement range");
  llvm::replace(c, c.begin() + beginOff, c.begin() + endOff, vals);
}

bool ConstantAggregateBuilder::add(mlir::Attribute a, CharUnits offset,
                                   bool allowOverwrite) {
  // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
  mlir::TypedAttr c = mlir::dyn_cast<mlir::TypedAttr>(a);
  assert(c && "expected typed attribute");
  // Common case: appending to a layout.
  if (offset >= size) {
    CharUnits align = getAlignment(c);
    CharUnits alignedSize = size.alignTo(align);
    if (alignedSize > offset || offset.alignTo(align) != offset)
      naturalLayout = false;
    else if (alignedSize < offset) {
      elems.push_back(getPadding(offset - size));
      offsets.push_back(size);
    }
    elems.push_back(c);
    offsets.push_back(offset);
    size = offset + getSize(c);
    return true;
  }

  // Uncommon case: constant overlaps what we've already created.
  std::optional<size_t> firstElemToReplace = splitAt(offset);
  if (!firstElemToReplace)
    return false;

  CharUnits cSize = getSize(c);
  std::optional<size_t> lastElemToReplace = splitAt(offset + cSize);
  if (!lastElemToReplace)
    return false;

  assert((firstElemToReplace == lastElemToReplace || allowOverwrite) &&
         "unexpectedly overwriting field");

  replace(elems, *firstElemToReplace, *lastElemToReplace, {c});
  replace(offsets, *firstElemToReplace, *lastElemToReplace, {offset});
  size = std::max(size, offset + cSize);
  naturalLayout = false;
  return true;
}

bool ConstantAggregateBuilder::addBits(llvm::APInt bits, uint64_t offsetInBits,
                                       bool allowOverwrite) {
  const ASTContext &context = cgm.getASTContext();
  const uint64_t charWidth = cgm.getASTContext().getCharWidth();
  auto charTy = cgm.getBuilder().getUIntNTy(charWidth);
  // Offset of where we want the first bit to go within the bits of the
  // current char.
  unsigned offsetWithinChar = offsetInBits % charWidth;

  // We split bit-fields up into individual bytes. Walk over the bytes and
  // update them.
  for (CharUnits offsetInChars =
           context.toCharUnitsFromBits(offsetInBits - offsetWithinChar);
       /**/; ++offsetInChars) {
    // Number of bits we want to fill in this char.
    unsigned wantedBits =
        std::min((uint64_t)bits.getBitWidth(), charWidth - offsetWithinChar);

    // Get a char containing the bits we want in the right places. The other
    // bits have unspecified values.
    llvm::APInt bitsThisChar = bits;
    if (bitsThisChar.getBitWidth() < charWidth)
      bitsThisChar = bitsThisChar.zext(charWidth);
    if (cgm.getDataLayout().isBigEndian()) {
      // Figure out how much to shift by. We may need to left-shift if we have
      // less than one byte of Bits left.
      int shift = bits.getBitWidth() - charWidth + offsetWithinChar;
      if (shift > 0)
        bitsThisChar.lshrInPlace(shift);
      else if (shift < 0)
        bitsThisChar = bitsThisChar.shl(-shift);
    } else {
      bitsThisChar = bitsThisChar.shl(offsetWithinChar);
    }
    if (bitsThisChar.getBitWidth() > charWidth)
      bitsThisChar = bitsThisChar.trunc(charWidth);

    if (wantedBits == charWidth) {
      // Got a full byte: just add it directly.
      add(mlir::cir::IntAttr::get(charTy, bitsThisChar), offsetInChars,
          allowOverwrite);
    } else {
      // Partial byte: update the existing integer if there is one. If we
      // can't split out a 1-CharUnit range to update, then we can't add
      // these bits and fail the entire constant emission.
      std::optional<size_t> firstElemToUpdate = splitAt(offsetInChars);
      if (!firstElemToUpdate)
        return false;
      std::optional<size_t> lastElemToUpdate =
          splitAt(offsetInChars + CharUnits::One());
      if (!lastElemToUpdate)
        return false;
      assert(*lastElemToUpdate - *firstElemToUpdate < 2 &&
             "should have at most one element covering one byte");

      // Figure out which bits we want and discard the rest.
      llvm::APInt updateMask(charWidth, 0);
      if (cgm.getDataLayout().isBigEndian())
        updateMask.setBits(charWidth - offsetWithinChar - wantedBits,
                           charWidth - offsetWithinChar);
      else
        updateMask.setBits(offsetWithinChar, offsetWithinChar + wantedBits);
      bitsThisChar &= updateMask;
      bool isNull = false;
      if (*firstElemToUpdate < elems.size()) {
        auto firstEltToUpdate =
            dyn_cast<mlir::cir::IntAttr>(elems[*firstElemToUpdate]);
        isNull = firstEltToUpdate && firstEltToUpdate.isNullValue();
      }

      if (*firstElemToUpdate == *lastElemToUpdate || isNull) {
        // All existing bits are either zero or undef.
        add(cgm.getBuilder().getAttr<mlir::cir::IntAttr>(charTy, bitsThisChar),
            offsetInChars, /*AllowOverwrite*/ true);
      } else {
        mlir::cir::IntAttr ci =
            dyn_cast<mlir::cir::IntAttr>(elems[*firstElemToUpdate]);
        // In order to perform a partial update, we need the existing bitwise
        // value, which we can only extract for a constant int.
        // auto *CI = dyn_cast<llvm::ConstantInt>(ToUpdate);
        if (!ci)
          return false;
        // Because this is a 1-CharUnit range, the constant occupying it must
        // be exactly one CharUnit wide.
        assert(ci.getBitWidth() == charWidth && "splitAt failed");
        assert((!(ci.getValue() & updateMask) || allowOverwrite) &&
               "unexpectedly overwriting bitfield");
        bitsThisChar |= (ci.getValue() & ~updateMask);
        elems[*firstElemToUpdate] =
            cgm.getBuilder().getAttr<mlir::cir::IntAttr>(charTy, bitsThisChar);
      }
    }

    // Stop if we've added all the bits.
    if (wantedBits == bits.getBitWidth())
      break;

    // Remove the consumed bits from Bits.
    if (!cgm.getDataLayout().isBigEndian())
      bits.lshrInPlace(wantedBits);
    bits = bits.trunc(bits.getBitWidth() - wantedBits);

    // The remanining bits go at the start of the following bytes.
    offsetWithinChar = 0;
  }

  return true;
}

/// Returns a position within Elems and Offsets such that all elements
/// before the returned index end before Pos and all elements at or after
/// the returned index begin at or after Pos. Splits elements as necessary
/// to ensure this. Returns None if we find something we can't split.
std::optional<size_t> ConstantAggregateBuilder::splitAt(CharUnits pos) {
  if (pos >= size)
    return offsets.size();

  while (true) {
    auto *firstAfterPos = llvm::upper_bound(offsets, pos);
    if (firstAfterPos == offsets.begin())
      return 0;

    // If we already have an element starting at Pos, we're done.
    size_t lastAtOrBeforePosIndex = firstAfterPos - offsets.begin() - 1;
    if (offsets[lastAtOrBeforePosIndex] == pos)
      return lastAtOrBeforePosIndex;

    // We found an element starting before Pos. Check for overlap.
    // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
    mlir::TypedAttr c =
        mlir::dyn_cast<mlir::TypedAttr>(elems[lastAtOrBeforePosIndex]);
    assert(c && "expected typed attribute");
    if (offsets[lastAtOrBeforePosIndex] + getSize(c) <= pos)
      return lastAtOrBeforePosIndex + 1;

    // Try to decompose it into smaller constants.
    if (!split(lastAtOrBeforePosIndex, pos))
      return std::nullopt;
  }
}

/// Split the constant at index Index, if possible. Return true if we did.
/// Hint indicates the location at which we'd like to split, but may be
/// ignored.
bool ConstantAggregateBuilder::split(size_t index, CharUnits hint) {
  llvm_unreachable("NYI");
}

mlir::Attribute ConstantAggregateBuilder::buildFrom(
    CIRGenModule &cgm, ArrayRef<mlir::Attribute> elems,
    ArrayRef<CharUnits> offsets, CharUnits startOffset, CharUnits size,
    bool naturalLayout, mlir::Type desiredTy, bool allowOversized) {
  ConstantAggregateBuilderUtils utils(cgm);

  if (elems.empty())
    return {};
  auto offset = [&](size_t i) { return offsets[i] - startOffset; };

  // If we want an array type, see if all the elements are the same type and
  // appropriately spaced.
  if (auto aty = mlir::dyn_cast<mlir::cir::ArrayType>(desiredTy)) {
    llvm_unreachable("NYI");
  }

  // The size of the constant we plan to generate. This is usually just the size
  // of the initialized type, but in AllowOversized mode (i.e. flexible array
  // init), it can be larger.
  CharUnits desiredSize = utils.getSize(desiredTy);
  if (size > desiredSize) {
    assert(allowOversized && "Elems are oversized");
    desiredSize = size;
  }

  // The natural alignment of an unpacked CIR struct with the given elements.
  CharUnits align = CharUnits::One();
  for (auto e : elems) {
    // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
    auto c = mlir::dyn_cast<mlir::TypedAttr>(e);
    assert(c && "expected typed attribute");
    align = std::max(align, utils.getAlignment(c));
  }

  // The natural size of an unpacked LLVM struct with the given elements.
  CharUnits alignedSize = size.alignTo(align);

  bool packed = false;
  ArrayRef<mlir::Attribute> unpackedElems = elems;
  llvm::SmallVector<mlir::Attribute, 32> unpackedElemStorage;
  if (desiredSize < alignedSize || desiredSize.alignTo(align) != desiredSize) {
    naturalLayout = false;
    packed = true;
  } else if (desiredSize > alignedSize) {
    // The natural layout would be too small. Add padding to fix it. (This
    // is ignored if we choose a packed layout.)
    unpackedElemStorage.assign(elems.begin(), elems.end());
    unpackedElemStorage.push_back(utils.getPadding(desiredSize - size));
    unpackedElems = unpackedElemStorage;
  }

  // If we don't have a natural layout, insert padding as necessary.
  // As we go, double-check to see if we can actually just emit Elems
  // as a non-packed struct and do so opportunistically if possible.
  llvm::SmallVector<mlir::Attribute, 32> packedElems;
  if (!naturalLayout) {
    CharUnits sizeSoFar = CharUnits::Zero();
    for (size_t i = 0; i != elems.size(); ++i) {
      mlir::TypedAttr c = mlir::dyn_cast<mlir::TypedAttr>(elems[i]);
      assert(c && "expected typed attribute");

      CharUnits align = utils.getAlignment(c);
      CharUnits naturalOffset = sizeSoFar.alignTo(align);
      CharUnits desiredOffset = offset(i);
      assert(desiredOffset >= sizeSoFar && "elements out of order");

      if (desiredOffset != naturalOffset)
        packed = true;
      if (desiredOffset != sizeSoFar)
        packedElems.push_back(utils.getPadding(desiredOffset - sizeSoFar));
      packedElems.push_back(elems[i]);
      sizeSoFar = desiredOffset + utils.getSize(c);
    }
    // If we're using the packed layout, pad it out to the desired size if
    // necessary.
    if (packed) {
      assert(sizeSoFar <= desiredSize &&
             "requested size is too small for contents");

      if (sizeSoFar < desiredSize)
        packedElems.push_back(utils.getPadding(desiredSize - sizeSoFar));
    }
  }

  auto &builder = cgm.getBuilder();
  auto arrAttr = mlir::ArrayAttr::get(builder.getContext(),
                                      packed ? packedElems : unpackedElems);
  auto strType = builder.getCompleteStructType(arrAttr, packed);

  if (auto desired = dyn_cast<mlir::cir::StructType>(desiredTy))
    if (desired.isLayoutIdentical(strType))
      strType = desired;

  return builder.getConstStructOrZeroAttr(arrAttr, packed, strType);
}

void ConstantAggregateBuilder::condense(CharUnits offset,
                                        mlir::Type desiredTy) {
  CharUnits size = getSize(desiredTy);

  std::optional<size_t> firstElemToReplace = splitAt(offset);
  if (!firstElemToReplace)
    return;
  size_t first = *firstElemToReplace;

  std::optional<size_t> lastElemToReplace = splitAt(offset + size);
  if (!lastElemToReplace)
    return;
  size_t last = *lastElemToReplace;

  size_t length = last - first;
  if (length == 0)
    return;

  // FIXME(cir): migrate most of this file to use mlir::TypedAttr directly.
  mlir::TypedAttr c = mlir::dyn_cast<mlir::TypedAttr>(elems[first]);
  assert(c && "expected typed attribute");
  if (length == 1 && offsets[first] == offset && getSize(c) == size) {
    // Re-wrap single element structs if necessary. Otherwise, leave any single
    // element constant of the right size alone even if it has the wrong type.
    llvm_unreachable("NYI");
  }

  mlir::Attribute replacement = buildFrom(
      cgm, ArrayRef(elems).slice(first, length),
      ArrayRef(offsets).slice(first, length), offset, getSize(desiredTy),
      /*known to have natural layout=*/false, desiredTy, false);
  replace(elems, first, last, {replacement});
  replace(offsets, first, last, {offset});
}

//===----------------------------------------------------------------------===//
//                            ConstStructBuilder
//===----------------------------------------------------------------------===//

class ConstStructBuilder {
  CIRGenModule &cgm;
  ConstantEmitter &emitter;
  ConstantAggregateBuilder &builder;
  CharUnits startOffset;

public:
  static mlir::Attribute buildStruct(ConstantEmitter &emitter,
                                     InitListExpr *ile, QualType structTy);
  static mlir::Attribute buildStruct(ConstantEmitter &emitter,
                                     const APValue &value, QualType valTy);
  static bool updateStruct(ConstantEmitter &emitter,
                           ConstantAggregateBuilder &Const, CharUnits offset,
                           InitListExpr *updater);

private:
  ConstStructBuilder(ConstantEmitter &emitter,
                     ConstantAggregateBuilder &builder, CharUnits startOffset)
      : cgm(emitter.CGM), emitter(emitter), builder(builder),
        startOffset(startOffset) {}

  bool appendField(const FieldDecl *field, uint64_t fieldOffset,
                   mlir::Attribute initExpr, bool allowOverwrite = false);

  bool appendBytes(CharUnits fieldOffsetInChars, mlir::Attribute initCst,
                   bool allowOverwrite = false);

  bool appendBitField(const FieldDecl *field, uint64_t fieldOffset,
                      mlir::cir::IntAttr initExpr, bool allowOverwrite = false);

  bool build(InitListExpr *ile, bool allowOverwrite);
  bool build(const APValue &val, const RecordDecl *rd, bool isPrimaryBase,
             const CXXRecordDecl *vTableClass, CharUnits baseOffset);
  mlir::Attribute finalize(QualType ty);
};

bool ConstStructBuilder::appendField(const FieldDecl *field,
                                     uint64_t fieldOffset,
                                     mlir::Attribute initCst,
                                     bool allowOverwrite) {
  const ASTContext &context = cgm.getASTContext();

  CharUnits fieldOffsetInChars = context.toCharUnitsFromBits(fieldOffset);

  return appendBytes(fieldOffsetInChars, initCst, allowOverwrite);
}

bool ConstStructBuilder::appendBytes(CharUnits fieldOffsetInChars,
                                     mlir::Attribute initCst,
                                     bool allowOverwrite) {
  return builder.add(initCst, startOffset + fieldOffsetInChars, allowOverwrite);
}

bool ConstStructBuilder::appendBitField(const FieldDecl *field,
                                        uint64_t fieldOffset,
                                        mlir::cir::IntAttr ci,
                                        bool allowOverwrite) {
  const auto &rl = cgm.getTypes().getCIRGenRecordLayout(field->getParent());
  const auto &info = rl.getBitFieldInfo(field);
  llvm::APInt fieldValue = ci.getValue();

  // Promote the size of FieldValue if necessary
  // FIXME: This should never occur, but currently it can because initializer
  // constants are cast to bool, and because clang is not enforcing bitfield
  // width limits.
  if (info.Size > fieldValue.getBitWidth())
    fieldValue = fieldValue.zext(info.Size);

  // Truncate the size of FieldValue to the bit field size.
  if (info.Size < fieldValue.getBitWidth())
    fieldValue = fieldValue.trunc(info.Size);

  return builder.addBits(fieldValue,
                         cgm.getASTContext().toBits(startOffset) + fieldOffset,
                         allowOverwrite);
}

static bool emitDesignatedInitUpdater(ConstantEmitter &emitter,
                                      ConstantAggregateBuilder &Const,
                                      CharUnits offset, QualType type,
                                      InitListExpr *updater) {
  if (type->isRecordType())
    return ConstStructBuilder::updateStruct(emitter, Const, offset, updater);

  const auto *cat = emitter.CGM.getASTContext().getAsConstantArrayType(type);
  if (!cat)
    return false;
  QualType elemType = cat->getElementType();
  CharUnits elemSize = emitter.CGM.getASTContext().getTypeSizeInChars(elemType);
  mlir::Type elemTy = emitter.CGM.getTypes().convertTypeForMem(elemType);

  mlir::Attribute fillC = nullptr;
  if (Expr *filler = updater->getArrayFiller()) {
    if (!isa<NoInitExpr>(filler)) {
      llvm_unreachable("NYI");
    }
  }

  unsigned numElementsToUpdate =
      fillC ? cat->getSize().getZExtValue() : updater->getNumInits();
  for (unsigned i = 0; i != numElementsToUpdate; ++i, offset += elemSize) {
    Expr *init = nullptr;
    if (i < updater->getNumInits())
      init = updater->getInit(i);

    if (!init && fillC) {
      if (!Const.add(fillC, offset, true))
        return false;
    } else if (!init || isa<NoInitExpr>(init)) {
      continue;
    } else if (InitListExpr *childIle = dyn_cast<InitListExpr>(init)) {
      if (!emitDesignatedInitUpdater(emitter, Const, offset, elemType,
                                     childIle))
        return false;
      // Attempt to reduce the array element to a single constant if necessary.
      Const.condense(offset, elemTy);
    } else {
      mlir::Attribute val = emitter.tryEmitPrivateForMemory(init, elemType);
      if (!Const.add(val, offset, true))
        return false;
    }
  }

  return true;
}

bool ConstStructBuilder::build(InitListExpr *ile, bool allowOverwrite) {
  RecordDecl *rd = ile->getType()->castAs<RecordType>()->getDecl();
  const ASTRecordLayout &layout = cgm.getASTContext().getASTRecordLayout(rd);

  unsigned fieldNo = -1;
  unsigned elementNo = 0;

  // Bail out if we have base classes. We could support these, but they only
  // arise in C++1z where we will have already constant folded most interesting
  // cases. FIXME: There are still a few more cases we can handle this way.
  if (auto *cxxrd = dyn_cast<CXXRecordDecl>(rd))
    if (cxxrd->getNumBases())
      return false;

  for (FieldDecl *field : rd->fields()) {
    ++fieldNo;

    // If this is a union, skip all the fields that aren't being initialized.
    if (rd->isUnion() &&
        !declaresSameEntity(ile->getInitializedFieldInUnion(), field))
      continue;

    // Don't emit anonymous bitfields.
    if (field->isUnnamedBitField())
      continue;

    // Get the initializer.  A struct can include fields without initializers,
    // we just use explicit null values for them.
    Expr *Init = nullptr;
    if (elementNo < ile->getNumInits())
      Init = ile->getInit(elementNo++);
    if (isa_and_nonnull<NoInitExpr>(Init))
      continue;

    // Zero-sized fields are not emitted, but their initializers may still
    // prevent emission of this struct as a constant.
    if (field->isZeroSize(cgm.getASTContext())) {
      if (Init->HasSideEffects(cgm.getASTContext()))
        return false;
      continue;
    }

    // When emitting a DesignatedInitUpdateExpr, a nested InitListExpr
    // represents additional overwriting of our current constant value, and not
    // a new constant to emit independently.
    if (allowOverwrite &&
        (field->getType()->isArrayType() || field->getType()->isRecordType())) {
      if (auto *subIle = dyn_cast<InitListExpr>(Init)) {
        CharUnits offset = cgm.getASTContext().toCharUnitsFromBits(
            layout.getFieldOffset(fieldNo));
        if (!emitDesignatedInitUpdater(emitter, builder, startOffset + offset,
                                       field->getType(), subIle))
          return false;
        // If we split apart the field's value, try to collapse it down to a
        // single value now.
        llvm_unreachable("NYI");
        continue;
      }
    }

    mlir::Attribute eltInit;
    if (Init)
      eltInit = emitter.tryEmitPrivateForMemory(Init, field->getType());
    else
      eltInit = emitter.emitNullForMemory(cgm.getLoc(ile->getSourceRange()),
                                          field->getType());

    if (!eltInit)
      return false;

    if (!field->isBitField()) {
      // Handle non-bitfield members.
      if (!appendField(field, layout.getFieldOffset(fieldNo), eltInit,
                       allowOverwrite))
        return false;
      // After emitting a non-empty field with [[no_unique_address]], we may
      // need to overwrite its tail padding.
      if (field->hasAttr<NoUniqueAddressAttr>())
        allowOverwrite = true;
    } else {
      // Otherwise we have a bitfield.
      if (auto constInt = dyn_cast<mlir::cir::IntAttr>(eltInit)) {
        if (!appendBitField(field, layout.getFieldOffset(fieldNo), constInt,
                            allowOverwrite))
          return false;
      } else {
        // We are trying to initialize a bitfield with a non-trivial constant,
        // this must require run-time code.
        return false;
      }
    }
  }

  return true;
}

namespace {
struct BaseInfo {
  BaseInfo(const CXXRecordDecl *decl, CharUnits offset, unsigned index)
      : decl(decl), offset(offset), index(index) {}

  const CXXRecordDecl *decl;
  CharUnits offset;
  unsigned index;

  bool operator<(const BaseInfo &o) const { return offset < o.offset; }
};
} // namespace

bool ConstStructBuilder::build(const APValue &val, const RecordDecl *rd,
                               bool isPrimaryBase,
                               const CXXRecordDecl *vTableClass,
                               CharUnits offset) {
  const ASTRecordLayout &layout = cgm.getASTContext().getASTRecordLayout(rd);

  if (const CXXRecordDecl *cd = dyn_cast<CXXRecordDecl>(rd)) {
    // Add a vtable pointer, if we need one and it hasn't already been added.
    if (layout.hasOwnVFPtr())
      llvm_unreachable("NYI");

    // Accumulate and sort bases, in order to visit them in address order, which
    // may not be the same as declaration order.
    SmallVector<BaseInfo, 8> Bases;
    Bases.reserve(cd->getNumBases());
    unsigned baseNo = 0;
    for (CXXRecordDecl::base_class_const_iterator base = cd->bases_begin(),
                                                  baseEnd = cd->bases_end();
         base != baseEnd; ++base, ++baseNo) {
      assert(!base->isVirtual() && "should not have virtual bases here");
      const CXXRecordDecl *bd = base->getType()->getAsCXXRecordDecl();
      CharUnits baseOffset = layout.getBaseClassOffset(bd);
      Bases.push_back(BaseInfo(bd, baseOffset, baseNo));
    }
    llvm::stable_sort(Bases);

    for (auto &Base : Bases) {
      bool isPrimaryBase = layout.getPrimaryBase() == Base.decl;
      build(val.getStructBase(Base.index), Base.decl, isPrimaryBase,
            vTableClass, offset + Base.offset);
    }
  }

  unsigned fieldNo = 0;
  uint64_t offsetBits = cgm.getASTContext().toBits(offset);

  bool allowOverwrite = false;
  for (RecordDecl::field_iterator field = rd->field_begin(),
                                  fieldEnd = rd->field_end();
       field != fieldEnd; ++field, ++fieldNo) {
    // If this is a union, skip all the fields that aren't being initialized.
    if (rd->isUnion() && !declaresSameEntity(val.getUnionField(), *field))
      continue;

    // Don't emit anonymous bitfields or zero-sized fields.
    if (field->isUnnamedBitField() || field->isZeroSize(cgm.getASTContext()))
      continue;

    // Emit the value of the initializer.
    const APValue &fieldValue =
        rd->isUnion() ? val.getUnionValue() : val.getStructField(fieldNo);
    mlir::Attribute eltInit =
        emitter.tryEmitPrivateForMemory(fieldValue, field->getType());
    if (!eltInit)
      return false;

    if (!field->isBitField()) {
      // Handle non-bitfield members.
      if (!appendField(*field, layout.getFieldOffset(fieldNo) + offsetBits,
                       eltInit, allowOverwrite))
        return false;
      // After emitting a non-empty field with [[no_unique_address]], we may
      // need to overwrite its tail padding.
      if (field->hasAttr<NoUniqueAddressAttr>())
        allowOverwrite = true;
    } else {
      llvm_unreachable("NYI");
    }
  }

  return true;
}

mlir::Attribute ConstStructBuilder::finalize(QualType type) {
  type = type.getNonReferenceType();
  RecordDecl *rd = type->castAs<RecordType>()->getDecl();
  mlir::Type valTy = cgm.getTypes().ConvertType(type);
  return builder.build(valTy, rd->hasFlexibleArrayMember());
}

mlir::Attribute ConstStructBuilder::buildStruct(ConstantEmitter &emitter,
                                                InitListExpr *ile,
                                                QualType valTy) {
  ConstantAggregateBuilder Const(emitter.CGM);
  ConstStructBuilder builder(emitter, Const, CharUnits::Zero());

  if (!builder.build(ile, /*AllowOverwrite*/ false))
    return nullptr;

  return builder.finalize(valTy);
}

mlir::Attribute ConstStructBuilder::buildStruct(ConstantEmitter &emitter,
                                                const APValue &val,
                                                QualType valTy) {
  ConstantAggregateBuilder Const(emitter.CGM);
  ConstStructBuilder builder(emitter, Const, CharUnits::Zero());

  const RecordDecl *rd = valTy->castAs<RecordType>()->getDecl();
  const CXXRecordDecl *cd = dyn_cast<CXXRecordDecl>(rd);
  if (!builder.build(val, rd, false, cd, CharUnits::Zero()))
    return nullptr;

  return builder.finalize(valTy);
}

bool ConstStructBuilder::updateStruct(ConstantEmitter &emitter,
                                      ConstantAggregateBuilder &Const,
                                      CharUnits offset, InitListExpr *updater) {
  return ConstStructBuilder(emitter, Const, offset)
      .build(updater, /*AllowOverwrite*/ true);
}

//===----------------------------------------------------------------------===//
//                             ConstExprEmitter
//===----------------------------------------------------------------------===//

// This class only needs to handle arrays, structs and unions.
//
// In LLVM codegen, when outside C++11 mode, those types are not constant
// folded, while all other types are handled by constant folding.
//
// In CIR codegen, instead of folding things here, we should defer that work
// to MLIR: do not attempt to do much here.
class ConstExprEmitter
    : public StmtVisitor<ConstExprEmitter, mlir::Attribute, QualType> {
  CIRGenModule &cgm;
  LLVM_ATTRIBUTE_UNUSED ConstantEmitter &emitter;

public:
  ConstExprEmitter(ConstantEmitter &emitter)
      : cgm(emitter.CGM), emitter(emitter) {}

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Attribute VisitStmt(Stmt *s, QualType t) { return nullptr; }

  mlir::Attribute VisitConstantExpr(ConstantExpr *ce, QualType t) {
    if (mlir::Attribute result = emitter.tryEmitConstantExpr(ce))
      return result;
    return Visit(ce->getSubExpr(), t);
  }

  mlir::Attribute VisitParenExpr(ParenExpr *pe, QualType t) {
    return Visit(pe->getSubExpr(), t);
  }

  mlir::Attribute
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *pe,
                                    QualType t) {
    return Visit(pe->getReplacement(), t);
  }

  mlir::Attribute VisitGenericSelectionExpr(GenericSelectionExpr *ge,
                                            QualType t) {
    return Visit(ge->getResultExpr(), t);
  }

  mlir::Attribute VisitChooseExpr(ChooseExpr *ce, QualType t) {
    return Visit(ce->getChosenSubExpr(), t);
  }

  mlir::Attribute VisitCompoundLiteralExpr(CompoundLiteralExpr *e, QualType t) {
    return Visit(e->getInitializer(), t);
  }

  mlir::Attribute VisitCastExpr(CastExpr *e, QualType destType) {
    if (const auto *ece = dyn_cast<ExplicitCastExpr>(e))
      cgm.buildExplicitCastExprType(ece, emitter.CGF);
    Expr *subExpr = e->getSubExpr();

    switch (e->getCastKind()) {
    case CK_HLSLArrayRValue:
    case CK_HLSLVectorTruncation:
    case CK_ToUnion:
      llvm_unreachable("not implemented");

    case CK_AddressSpaceConversion: {
      llvm_unreachable("not implemented");
    }

    case CK_LValueToRValue:
    case CK_AtomicToNonAtomic:
    case CK_NonAtomicToAtomic:
    case CK_NoOp:
    case CK_ConstructorConversion:
      return Visit(subExpr, destType);

    case CK_IntToOCLSampler:
      llvm_unreachable("global sampler variables are not generated");

    case CK_Dependent:
      llvm_unreachable("saw dependent cast!");

    case CK_BuiltinFnToFnPtr:
      llvm_unreachable("builtin functions are handled elsewhere");

    case CK_ReinterpretMemberPointer:
    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer: {
      llvm_unreachable("not implemented");
    }

    // These will never be supported.
    case CK_ObjCObjectLValueCast:
    case CK_ARCProduceObject:
    case CK_ARCConsumeObject:
    case CK_ARCReclaimReturnedObject:
    case CK_ARCExtendBlockObject:
    case CK_CopyAndAutoreleaseBlockObject:
      return nullptr;

    // These don't need to be handled here because Evaluate knows how to
    // evaluate them in the cases where they can be folded.
    case CK_BitCast:
    case CK_ToVoid:
    case CK_Dynamic:
    case CK_LValueBitCast:
    case CK_LValueToRValueBitCast:
    case CK_NullToMemberPointer:
    case CK_UserDefinedConversion:
    case CK_CPointerToObjCPointerCast:
    case CK_BlockPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
    case CK_ArrayToPointerDecay:
    case CK_FunctionToPointerDecay:
    case CK_BaseToDerived:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_MemberPointerToBoolean:
    case CK_VectorSplat:
    case CK_FloatingRealToComplex:
    case CK_FloatingComplexToReal:
    case CK_FloatingComplexToBoolean:
    case CK_FloatingComplexCast:
    case CK_FloatingComplexToIntegralComplex:
    case CK_IntegralRealToComplex:
    case CK_IntegralComplexToReal:
    case CK_IntegralComplexToBoolean:
    case CK_IntegralComplexCast:
    case CK_IntegralComplexToFloatingComplex:
    case CK_PointerToIntegral:
    case CK_PointerToBoolean:
    case CK_NullToPointer:
    case CK_IntegralCast:
    case CK_BooleanToSignedIntegral:
    case CK_IntegralToPointer:
    case CK_IntegralToBoolean:
    case CK_IntegralToFloating:
    case CK_FloatingToIntegral:
    case CK_FloatingToBoolean:
    case CK_FloatingCast:
    case CK_FloatingToFixedPoint:
    case CK_FixedPointToFloating:
    case CK_FixedPointCast:
    case CK_FixedPointToBoolean:
    case CK_FixedPointToIntegral:
    case CK_IntegralToFixedPoint:
    case CK_ZeroToOCLOpaqueType:
    case CK_MatrixCast:
      return nullptr;
    }
    llvm_unreachable("Invalid CastKind");
  }

  mlir::Attribute VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die, QualType t) {
    // TODO(cir): figure out CIR story here...
    // No need for a DefaultInitExprScope: we don't handle 'this' in a
    // constant expression.
    return Visit(die->getExpr(), t);
  }

  mlir::Attribute VisitExprWithCleanups(ExprWithCleanups *e, QualType t) {
    // Since this about constant emission no need to wrap this under a scope.
    return Visit(e->getSubExpr(), t);
  }

  mlir::Attribute VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *e,
                                                QualType t) {
    return Visit(e->getSubExpr(), t);
  }

  mlir::Attribute emitArrayInitialization(InitListExpr *ile, QualType t) {
    auto *cat = cgm.getASTContext().getAsConstantArrayType(ile->getType());
    assert(cat && "can't emit array init for non-constant-bound array");
    unsigned numInitElements = ile->getNumInits();        // init list size
    unsigned numElements = cat->getSize().getZExtValue(); // array size
    unsigned numInitableElts = std::min(numInitElements, numElements);

    QualType eltTy = cat->getElementType();
    SmallVector<mlir::TypedAttr, 16> elts;
    elts.reserve(numElements);

    // Emit array filler, if there is one.
    mlir::Attribute filler;
    if (ile->hasArrayFiller()) {
      auto *aux = ile->getArrayFiller();
      filler = emitter.tryEmitAbstractForMemory(aux, cat->getElementType());
      if (!filler)
        return {};
    }

    // Emit initializer elements as MLIR attributes and check for common type.
    mlir::Type commonElementType;
    for (unsigned i = 0; i != numInitableElts; ++i) {
      Expr *init = ile->getInit(i);
      auto c = emitter.tryEmitPrivateForMemory(init, eltTy);
      if (!c)
        return {};
      if (i == 0)
        commonElementType = c.getType();
      else if (c.getType() != commonElementType)
        commonElementType = nullptr;
      elts.push_back(c);
    }

    auto desiredType = cgm.getTypes().ConvertType(t);
    auto typedFiller = llvm::dyn_cast_or_null<mlir::TypedAttr>(filler);
    if (filler && !typedFiller)
      llvm_unreachable("We shouldn't be receiving untyped attrs here");
    return buildArrayConstant(cgm, desiredType, commonElementType, numElements,
                              elts, typedFiller);
  }

  mlir::Attribute emitRecordInitialization(InitListExpr *ile, QualType t) {
    return ConstStructBuilder::buildStruct(emitter, ile, t);
  }

  mlir::Attribute emitVectorInitialization(InitListExpr *ile, QualType t) {
    mlir::cir::VectorType vecTy =
        mlir::cast<mlir::cir::VectorType>(cgm.getTypes().ConvertType(t));
    unsigned numElements = vecTy.getSize();
    unsigned numInits = ile->getNumInits();
    assert(numElements >= numInits && "Too many initializers for a vector");
    QualType eltTy = t->castAs<VectorType>()->getElementType();
    SmallVector<mlir::Attribute, 8> elts;
    // Process the explicit initializers
    for (unsigned i = 0; i < numInits; ++i) {
      auto value = emitter.tryEmitPrivateForMemory(ile->getInit(i), eltTy);
      if (!value)
        return {};
      elts.push_back(value);
    }
    // Zero-fill the rest of the vector
    for (unsigned i = numInits; i < numElements; ++i) {
      elts.push_back(cgm.getBuilder().getZeroInitAttr(vecTy.getEltType()));
    }
    return mlir::cir::ConstVectorAttr::get(
        vecTy, mlir::ArrayAttr::get(cgm.getBuilder().getContext(), elts));
  }

  mlir::Attribute VisitImplicitValueInitExpr(ImplicitValueInitExpr *e,
                                             QualType t) {
    return cgm.getBuilder().getZeroInitAttr(cgm.getCIRType(t));
  }

  mlir::Attribute VisitInitListExpr(InitListExpr *ile, QualType t) {
    if (ile->isTransparent())
      return Visit(ile->getInit(0), t);

    if (ile->getType()->isArrayType())
      return emitArrayInitialization(ile, t);

    if (ile->getType()->isRecordType())
      return emitRecordInitialization(ile, t);

    if (ile->getType()->isVectorType())
      return emitVectorInitialization(ile, t);

    return nullptr;
  }

  mlir::Attribute VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *e,
                                                QualType destType) {
    auto c = Visit(e->getBase(), destType);
    if (!c)
      return nullptr;

    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitCXXConstructExpr(CXXConstructExpr *e, QualType ty) {
    if (!e->getConstructor()->isTrivial())
      return nullptr;

    // Only default and copy/move constructors can be trivial.
    if (e->getNumArgs()) {
      assert(e->getNumArgs() == 1 && "trivial ctor with > 1 argument");
      assert(e->getConstructor()->isCopyOrMoveConstructor() &&
             "trivial ctor has argument but isn't a copy/move ctor");

      Expr *arg = e->getArg(0);
      assert(cgm.getASTContext().hasSameUnqualifiedType(ty, arg->getType()) &&
             "argument to copy ctor is of wrong type");

      // Look through the temporary; it's just converting the value to an lvalue
      // to pass it to the constructor.
      if (auto *mte = dyn_cast<MaterializeTemporaryExpr>(arg))
        return Visit(mte->getSubExpr(), ty);
      // Don't try to support arbitrary lvalue-to-rvalue conversions for now.
      return nullptr;
    }

    return cgm.getBuilder().getZeroInitAttr(cgm.getCIRType(ty));
  }

  mlir::Attribute VisitStringLiteral(StringLiteral *e, QualType t) {
    // This is a string literal initializing an array in an initializer.
    return cgm.getConstantArrayFromStringLiteral(e);
  }

  mlir::Attribute VisitObjCEncodeExpr(ObjCEncodeExpr *e, QualType t) {
    assert(0 && "not implemented");
    return {};
  }

  mlir::Attribute VisitUnaryExtension(const UnaryOperator *e, QualType t) {
    return Visit(e->getSubExpr(), t);
  }

  // Utility methods
  mlir::Type convertType(QualType t) { return cgm.getTypes().ConvertType(t); }
};

static mlir::Attribute
buildArrayConstant(CIRGenModule &cgm, mlir::Type desiredType,
                   mlir::Type CommonElementType, unsigned ArrayBound,
                   SmallVectorImpl<mlir::TypedAttr> &Elements,
                   mlir::TypedAttr Filler) {
  auto &builder = cgm.getBuilder();

  // Figure out how long the initial prefix of non-zero elements is.
  unsigned nonzeroLength = ArrayBound;
  if (Elements.size() < nonzeroLength && builder.isNullValue(Filler))
    nonzeroLength = Elements.size();
  if (nonzeroLength == Elements.size()) {
    while (nonzeroLength > 0 &&
           builder.isNullValue(Elements[nonzeroLength - 1]))
      --nonzeroLength;
  }

  if (nonzeroLength == 0)
    return builder.getZeroInitAttr(desiredType);

  // Add a zeroinitializer array filler if we have lots of trailing zeroes.
  unsigned trailingZeroes = ArrayBound - nonzeroLength;
  if (trailingZeroes >= 8) {
    assert(Elements.size() >= nonzeroLength &&
           "missing initializer for non-zero element");

    SmallVector<mlir::Attribute, 4> eles;
    eles.reserve(Elements.size());
    for (auto const &element : Elements)
      eles.push_back(element);

    return builder.getConstArray(
        mlir::ArrayAttr::get(builder.getContext(), eles),
        mlir::cir::ArrayType::get(builder.getContext(), CommonElementType,
                                  ArrayBound));
    // TODO(cir): If all the elements had the same type up to the trailing
    // zeroes, emit a struct of two arrays (the nonzero data and the
    // zeroinitializer). Use DesiredType to get the element type.
  }
  if (Elements.size() != ArrayBound) {
    // Otherwise pad to the right size with the filler if necessary.
    Elements.resize(ArrayBound, Filler);
    if (Filler.getType() != CommonElementType)
      CommonElementType = {};
  }

  // If all elements have the same type, just emit an array constant.
  if (CommonElementType) {
    SmallVector<mlir::Attribute, 4> eles;
    eles.reserve(Elements.size());
    for (auto const &element : Elements)
      eles.push_back(element);

    return builder.getConstArray(
        mlir::ArrayAttr::get(builder.getContext(), eles),
        mlir::cir::ArrayType::get(builder.getContext(), CommonElementType,
                                  ArrayBound));
  }

  SmallVector<mlir::Attribute, 4> eles;
  eles.reserve(Elements.size());
  for (auto const &element : Elements)
    eles.push_back(element);

  auto arrAttr = mlir::ArrayAttr::get(builder.getContext(), eles);
  return builder.getAnonConstStruct(arrAttr, false);
}

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                          ConstantLValueEmitter
//===----------------------------------------------------------------------===//

namespace {
/// A struct which can be used to peephole certain kinds of finalization
/// that normally happen during l-value emission.
struct ConstantLValue {
  llvm::PointerUnion<mlir::Value, mlir::Attribute> value;
  bool hasOffsetApplied;

  /*implicit*/ ConstantLValue(mlir::Value value, bool hasOffsetApplied = false)
      : value(value), hasOffsetApplied(hasOffsetApplied) {}

  /*implicit*/ ConstantLValue(mlir::cir::GlobalViewAttr address)
      : value(address), hasOffsetApplied(false) {}

  ConstantLValue(std::nullptr_t) : ConstantLValue({}, false) {}
};

/// A helper class for emitting constant l-values.
class ConstantLValueEmitter
    : public ConstStmtVisitor<ConstantLValueEmitter, ConstantLValue> {
  CIRGenModule &cgm;
  ConstantEmitter &emitter;
  const APValue &value;
  QualType destType;

  // Befriend StmtVisitorBase so that we don't have to expose Visit*.
  friend StmtVisitorBase;

public:
  ConstantLValueEmitter(ConstantEmitter &emitter, const APValue &value,
                        QualType destType)
      : cgm(emitter.CGM), emitter(emitter), value(value), destType(destType) {}

  mlir::Attribute tryEmit();

private:
  mlir::Attribute tryEmitAbsolute(mlir::Type destTy);
  ConstantLValue tryEmitBase(const APValue::LValueBase &base);

  ConstantLValue VisitStmt(const Stmt *s) { return nullptr; }
  ConstantLValue VisitConstantExpr(const ConstantExpr *e);
  ConstantLValue VisitCompoundLiteralExpr(const CompoundLiteralExpr *e);
  ConstantLValue VisitStringLiteral(const StringLiteral *e);
  ConstantLValue VisitObjCBoxedExpr(const ObjCBoxedExpr *e);
  ConstantLValue VisitObjCEncodeExpr(const ObjCEncodeExpr *e);
  ConstantLValue VisitObjCStringLiteral(const ObjCStringLiteral *e);
  ConstantLValue VisitPredefinedExpr(const PredefinedExpr *e);
  ConstantLValue VisitAddrLabelExpr(const AddrLabelExpr *e);
  ConstantLValue VisitCallExpr(const CallExpr *e);
  ConstantLValue VisitBlockExpr(const BlockExpr *e);
  ConstantLValue VisitCXXTypeidExpr(const CXXTypeidExpr *e);
  ConstantLValue
  VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *expr);

  bool hasNonZeroOffset() const { return !value.getLValueOffset().isZero(); }

  /// Return GEP-like value offset
  mlir::ArrayAttr getOffset(mlir::Type ty) {
    auto offset = value.getLValueOffset().getQuantity();
    CIRDataLayout layout(cgm.getModule());
    SmallVector<int64_t, 3> idx;
    cgm.getBuilder().computeGlobalViewIndicesFromFlatOffset(offset, ty, layout,
                                                            idx);

    llvm::SmallVector<mlir::Attribute, 3> indices;
    for (auto i : idx) {
      auto attr = cgm.getBuilder().getI32IntegerAttr(i);
      indices.push_back(attr);
    }

    if (indices.empty())
      return {};
    return cgm.getBuilder().getArrayAttr(indices);
  }

  // TODO(cir): create a proper interface to absctract CIR constant values.

  /// Apply the value offset to the given constant.
  ConstantLValue applyOffset(ConstantLValue &c) {

    // Handle attribute constant LValues.
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(c.value)) {
      if (auto gv = mlir::dyn_cast<mlir::cir::GlobalViewAttr>(attr)) {
        auto baseTy =
            mlir::cast<mlir::cir::PointerType>(gv.getType()).getPointee();
        auto destTy = cgm.getTypes().convertTypeForMem(destType);
        assert(!gv.getIndices() && "Global view is already indexed");
        return mlir::cir::GlobalViewAttr::get(destTy, gv.getSymbol(),
                                              getOffset(baseTy));
      }
      llvm_unreachable("Unsupported attribute type to offset");
    }

    // TODO(cir): use ptr_stride, or something...
    llvm_unreachable("NYI");
  }
};

} // namespace

mlir::Attribute ConstantLValueEmitter::tryEmit() {
  const APValue::LValueBase &base = value.getLValueBase();

  // The destination type should be a pointer or reference
  // type, but it might also be a cast thereof.
  //
  // FIXME: the chain of casts required should be reflected in the APValue.
  // We need this in order to correctly handle things like a ptrtoint of a
  // non-zero null pointer and addrspace casts that aren't trivially
  // represented in LLVM IR.
  auto destTy = cgm.getTypes().convertTypeForMem(destType);
  assert(mlir::isa<mlir::cir::PointerType>(destTy));

  // If there's no base at all, this is a null or absolute pointer,
  // possibly cast back to an integer type.
  if (!base) {
    return tryEmitAbsolute(destTy);
  }

  // Otherwise, try to emit the base.
  ConstantLValue result = tryEmitBase(base);

  // If that failed, we're done.
  auto &value = result.value;
  if (!value)
    return {};

  // Apply the offset if necessary and not already done.
  if (!result.hasOffsetApplied) {
    value = applyOffset(result).value;
  }

  // Convert to the appropriate type; this could be an lvalue for
  // an integer. FIXME: performAddrSpaceCast
  if (mlir::isa<mlir::cir::PointerType>(destTy)) {
    if (value.is<mlir::Attribute>())
      return value.get<mlir::Attribute>();
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
}

/// Try to emit an absolute l-value, such as a null pointer or an integer
/// bitcast to pointer type.
mlir::Attribute ConstantLValueEmitter::tryEmitAbsolute(mlir::Type destTy) {
  // If we're producing a pointer, this is easy.
  auto destPtrTy = mlir::dyn_cast<mlir::cir::PointerType>(destTy);
  assert(destPtrTy && "expected !cir.ptr type");
  return cgm.getBuilder().getConstPtrAttr(
      destPtrTy, value.getLValueOffset().getQuantity());
}

ConstantLValue
ConstantLValueEmitter::tryEmitBase(const APValue::LValueBase &base) {
  // Handle values.
  if (const ValueDecl *d = base.dyn_cast<const ValueDecl *>()) {
    // The constant always points to the canonical declaration. We want to look
    // at properties of the most recent declaration at the point of emission.
    d = cast<ValueDecl>(d->getMostRecentDecl());

    if (d->hasAttr<WeakRefAttr>())
      llvm_unreachable("emit pointer base for weakref is NYI");

    if (auto *fd = dyn_cast<FunctionDecl>(d)) {
      auto fop = cgm.GetAddrOfFunction(fd);
      auto builder = cgm.getBuilder();
      auto *ctxt = builder.getContext();
      return mlir::cir::GlobalViewAttr::get(
          builder.getPointerTo(fop.getFunctionType()),
          mlir::FlatSymbolRefAttr::get(ctxt, fop.getSymNameAttr()));
    }

    if (auto *vd = dyn_cast<VarDecl>(d)) {
      // We can never refer to a variable with local storage.
      if (!vd->hasLocalStorage()) {
        if (vd->isFileVarDecl() || vd->hasExternalStorage())
          return cgm.getAddrOfGlobalVarAttr(vd);

        if (vd->isLocalVarDecl()) {
          auto linkage =
              cgm.getCIRLinkageVarDefinition(vd, /*IsConstant=*/false);
          return cgm.getBuilder().getGlobalViewAttr(
              cgm.getOrCreateStaticVarDecl(*vd, linkage));
        }
      }
    }
  }

  // Handle typeid(T).
  if (TypeInfoLValue ti = base.dyn_cast<TypeInfoLValue>()) {
    assert(0 && "NYI");
  }

  // Otherwise, it must be an expression.
  return Visit(base.get<const Expr *>());
}

static ConstantLValue
tryEmitGlobalCompoundLiteral(ConstantEmitter &emitter,
                             const CompoundLiteralExpr *e) {
  CIRGenModule &cgm = emitter.CGM;

  LangAS addressSpace = e->getType().getAddressSpace();
  mlir::Attribute c = emitter.tryEmitForInitializer(e->getInitializer(),
                                                    addressSpace, e->getType());
  if (!c) {
    assert(!e->isFileScope() &&
           "file-scope compound literal did not have constant initializer!");
    return nullptr;
  }

  auto gv = CIRGenModule::createGlobalOp(
      cgm, cgm.getLoc(e->getSourceRange()),
      cgm.createGlobalCompoundLiteralName(),
      cgm.getTypes().convertTypeForMem(e->getType()),
      e->getType().isConstantStorage(cgm.getASTContext(), false, false));
  gv.setInitialValueAttr(c);
  gv.setLinkage(mlir::cir::GlobalLinkageKind::InternalLinkage);
  CharUnits align = cgm.getASTContext().getTypeAlignInChars(e->getType());
  gv.setAlignment(align.getAsAlign().value());

  emitter.finalize(gv);
  return cgm.getBuilder().getGlobalViewAttr(gv);
}

ConstantLValue ConstantLValueEmitter::VisitConstantExpr(const ConstantExpr *e) {
  assert(0 && "NYI");
  return Visit(e->getSubExpr());
}

ConstantLValue
ConstantLValueEmitter::VisitCompoundLiteralExpr(const CompoundLiteralExpr *e) {
  ConstantEmitter compoundLiteralEmitter(cgm, emitter.CGF);
  compoundLiteralEmitter.setInConstantContext(emitter.isInConstantContext());
  return tryEmitGlobalCompoundLiteral(compoundLiteralEmitter, e);
}

ConstantLValue
ConstantLValueEmitter::VisitStringLiteral(const StringLiteral *e) {
  return cgm.getAddrOfConstantStringFromLiteral(e);
}

ConstantLValue
ConstantLValueEmitter::VisitObjCEncodeExpr(const ObjCEncodeExpr *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitObjCStringLiteral(const ObjCStringLiteral *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitObjCBoxedExpr(const ObjCBoxedExpr *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitPredefinedExpr(const PredefinedExpr *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitAddrLabelExpr(const AddrLabelExpr *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue ConstantLValueEmitter::VisitCallExpr(const CallExpr *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue ConstantLValueEmitter::VisitBlockExpr(const BlockExpr *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue
ConstantLValueEmitter::VisitCXXTypeidExpr(const CXXTypeidExpr *e) {
  assert(0 && "NYI");
  return nullptr;
}

ConstantLValue ConstantLValueEmitter::VisitMaterializeTemporaryExpr(
    const MaterializeTemporaryExpr *expr) {
  assert(expr->getStorageDuration() == SD_Static);
  const Expr *inner = expr->getSubExpr()->skipRValueSubobjectAdjustments();
  mlir::Operation *globalTemp = cgm.getAddrOfGlobalTemporary(expr, inner);
  CIRGenBuilderTy builder = cgm.getBuilder();
  return ConstantLValue(
      builder.getGlobalViewAttr(mlir::cast<mlir::cir::GlobalOp>(globalTemp)));
}

//===----------------------------------------------------------------------===//
//                             ConstantEmitter
//===----------------------------------------------------------------------===//

mlir::Attribute ConstantEmitter::validateAndPopAbstract(mlir::Attribute c,
                                                        AbstractState saved) {
  Abstract = saved.OldValue;

  assert(saved.OldPlaceholdersSize == PlaceholderAddresses.size() &&
         "created a placeholder while doing an abstract emission?");

  // No validation necessary for now.
  // No cleanup to do for now.
  return c;
}

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const VarDecl &d) {
  initializeNonAbstract(d.getType().getAddressSpace());
  return markIfFailed(tryEmitPrivateForVarInit(d));
}

mlir::Attribute ConstantEmitter::tryEmitForInitializer(const Expr *e,
                                                       LangAS destAddrSpace,
                                                       QualType destType) {
  initializeNonAbstract(destAddrSpace);
  return markIfFailed(tryEmitPrivateForMemory(e, destType));
}

mlir::Attribute ConstantEmitter::emitForInitializer(const APValue &value,
                                                    LangAS destAddrSpace,
                                                    QualType destType) {
  initializeNonAbstract(destAddrSpace);
  auto c = tryEmitPrivateForMemory(value, destType);
  assert(c && "couldn't emit constant value non-abstractly?");
  return c;
}

void ConstantEmitter::finalize(mlir::cir::GlobalOp global) {
  assert(InitializedNonAbstract &&
         "finalizing emitter that was used for abstract emission?");
  assert(!Finalized && "finalizing emitter multiple times");
  assert(!global.isDeclaration());

  // Note that we might also be Failed.
  Finalized = true;

  if (!PlaceholderAddresses.empty()) {
    assert(0 && "not implemented");
  }
}

ConstantEmitter::~ConstantEmitter() {
  assert((!InitializedNonAbstract || Finalized || Failed) &&
         "not finalized after being initialized for non-abstract emission");
  assert(PlaceholderAddresses.empty() && "unhandled placeholders");
}

// TODO(cir): this can be shared with LLVM's codegen
static QualType getNonMemoryType(CIRGenModule &cgm, QualType type) {
  if (const auto *at = type->getAs<AtomicType>()) {
    return cgm.getASTContext().getQualifiedType(at->getValueType(),
                                                type.getQualifiers());
  }
  return type;
}

mlir::Attribute
ConstantEmitter::tryEmitAbstractForInitializer(const VarDecl &d) {
  auto state = pushAbstract();
  auto c = tryEmitPrivateForVarInit(d);
  return validateAndPopAbstract(c, state);
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForVarInit(const VarDecl &d) {
  // Make a quick check if variable can be default NULL initialized
  // and avoid going through rest of code which may do, for c++11,
  // initialization of memory to all NULLs.
  if (!d.hasLocalStorage()) {
    QualType ty = CGM.getASTContext().getBaseElementType(d.getType());
    if (ty->isRecordType())
      if (const CXXConstructExpr *e =
              dyn_cast_or_null<CXXConstructExpr>(d.getInit())) {
        const CXXConstructorDecl *cd = e->getConstructor();
        // FIXME: we should probably model this more closely to C++ than
        // just emitting a global with zero init (mimic what we do for trivial
        // assignments and whatnots). Since this is for globals shouldn't
        // be a problem for the near future.
        if (cd->isTrivial() && cd->isDefaultConstructor())
          return mlir::cir::ZeroAttr::get(
              CGM.getBuilder().getContext(),
              CGM.getTypes().ConvertType(d.getType()));
      }
  }
  InConstantContext = d.hasConstantInitialization();

  const Expr *e = d.getInit();
  assert(e && "No initializer to emit");

  QualType destType = d.getType();

  if (!destType->isReferenceType()) {
    QualType nonMemoryDestType = getNonMemoryType(CGM, destType);
    if (auto c = ConstExprEmitter(*this).Visit(const_cast<Expr *>(e),
                                               nonMemoryDestType))
      return emitForMemory(c, destType);
  }

  // Try to emit the initializer.  Note that this can allow some things that
  // are not allowed by tryEmitPrivateForMemory alone.
  if (auto *value = d.evaluateValue())
    return tryEmitPrivateForMemory(*value, destType);

  return nullptr;
}

mlir::Attribute ConstantEmitter::tryEmitAbstract(const Expr *e,
                                                 QualType destType) {
  auto state = pushAbstract();
  auto c = tryEmitPrivate(e, destType);
  return validateAndPopAbstract(c, state);
}

mlir::Attribute ConstantEmitter::tryEmitAbstract(const APValue &value,
                                                 QualType destType) {
  auto state = pushAbstract();
  auto c = tryEmitPrivate(value, destType);
  return validateAndPopAbstract(c, state);
}

mlir::Attribute ConstantEmitter::tryEmitConstantExpr(const ConstantExpr *ce) {
  if (!ce->hasAPValueResult())
    return nullptr;

  QualType retType = ce->getType();
  if (ce->isGLValue())
    retType = CGM.getASTContext().getLValueReferenceType(retType);

  return emitAbstract(ce->getBeginLoc(), ce->getAPValueResult(), retType);
}

mlir::Attribute ConstantEmitter::tryEmitAbstractForMemory(const Expr *e,
                                                          QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto c = tryEmitAbstract(e, nonMemoryDestType);
  return (c ? emitForMemory(c, destType) : nullptr);
}

mlir::Attribute ConstantEmitter::tryEmitAbstractForMemory(const APValue &value,
                                                          QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto c = tryEmitAbstract(value, nonMemoryDestType);
  return (c ? emitForMemory(c, destType) : nullptr);
}

mlir::TypedAttr ConstantEmitter::tryEmitPrivateForMemory(const Expr *e,
                                                         QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto c = tryEmitPrivate(e, nonMemoryDestType);
  if (c) {
    auto attr = emitForMemory(c, destType);
    auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(attr);
    if (!typedAttr)
      llvm_unreachable("this should always be typed");
    return typedAttr;
  }
  return nullptr;
}

mlir::Attribute ConstantEmitter::tryEmitPrivateForMemory(const APValue &value,
                                                         QualType destType) {
  auto nonMemoryDestType = getNonMemoryType(CGM, destType);
  auto c = tryEmitPrivate(value, nonMemoryDestType);
  return (c ? emitForMemory(c, destType) : nullptr);
}

mlir::Attribute ConstantEmitter::emitForMemory(CIRGenModule &cgm,
                                               mlir::Attribute c,
                                               QualType destType) {
  // For an _Atomic-qualified constant, we may need to add tail padding.
  if (const auto *at = destType->getAs<AtomicType>()) {
    QualType destValueType = at->getValueType();
    c = emitForMemory(cgm, c, destValueType);

    uint64_t innerSize = cgm.getASTContext().getTypeSize(destValueType);
    uint64_t outerSize = cgm.getASTContext().getTypeSize(destType);
    if (innerSize == outerSize)
      return c;

    assert(innerSize < outerSize && "emitted over-large constant for atomic");
    auto &builder = cgm.getBuilder();
    auto zeroArray = builder.getZeroInitAttr(
        mlir::cir::ArrayType::get(builder.getContext(), builder.getUInt8Ty(),
                                  (outerSize - innerSize) / 8));
    SmallVector<mlir::Attribute, 4> anonElts = {c, zeroArray};
    auto arrAttr = mlir::ArrayAttr::get(builder.getContext(), anonElts);
    return builder.getAnonConstStruct(arrAttr, false);
  }

  // Zero-extend bool.
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(c);
  if (typed && mlir::isa<mlir::cir::BoolType>(typed.getType())) {
    // Already taken care given that bool values coming from
    // integers only carry true/false.
  }

  return c;
}

mlir::TypedAttr ConstantEmitter::tryEmitPrivate(const Expr *e,
                                                QualType destType) {
  assert(!destType->isVoidType() && "can't emit a void constant");

  if (auto c = ConstExprEmitter(*this).Visit(const_cast<Expr *>(e), destType)) {
    if (auto typedC = mlir::dyn_cast_if_present<mlir::TypedAttr>(c))
      return typedC;
    llvm_unreachable("this should always be typed");
  }

  Expr::EvalResult result;

  bool success;

  if (destType->isReferenceType())
    success = e->EvaluateAsLValue(result, CGM.getASTContext());
  else
    success =
        e->EvaluateAsRValue(result, CGM.getASTContext(), InConstantContext);

  if (success && !result.hasSideEffects()) {
    auto c = tryEmitPrivate(result.Val, destType);
    if (auto typedC = mlir::dyn_cast_if_present<mlir::TypedAttr>(c))
      return typedC;
    llvm_unreachable("this should always be typed");
  }

  return nullptr;
}

mlir::Attribute ConstantEmitter::tryEmitPrivate(const APValue &value,
                                                QualType destType) {
  auto &builder = CGM.getBuilder();
  switch (value.getKind()) {
  case APValue::None:
  case APValue::Indeterminate:
    // TODO(cir): LLVM models out-of-lifetime and indeterminate values as
    // 'undef'. Find out what's better for CIR.
    assert(0 && "not implemented");
  case APValue::Int: {
    mlir::Type ty = CGM.getCIRType(destType);
    if (mlir::isa<mlir::cir::BoolType>(ty))
      return builder.getCIRBoolAttr(value.getInt().getZExtValue());
    assert(mlir::isa<mlir::cir::IntType>(ty) && "expected integral type");
    return CGM.getBuilder().getAttr<mlir::cir::IntAttr>(ty, value.getInt());
  }
  case APValue::Float: {
    const llvm::APFloat &init = value.getFloat();
    if (&init.getSemantics() == &llvm::APFloat::IEEEhalf() &&
        !CGM.getASTContext().getLangOpts().NativeHalfType &&
        CGM.getASTContext().getTargetInfo().useFP16ConversionIntrinsics())
      assert(0 && "not implemented");
    else {
      mlir::Type ty = CGM.getCIRType(destType);
      assert(mlir::isa<mlir::cir::CIRFPTypeInterface>(ty) &&
             "expected floating-point type");
      return CGM.getBuilder().getAttr<mlir::cir::FPAttr>(ty, init);
    }
  }
  case APValue::Array: {
    const ArrayType *arrayTy = CGM.getASTContext().getAsArrayType(destType);
    unsigned numElements = value.getArraySize();
    unsigned numInitElts = value.getArrayInitializedElts();

    // Emit array filler, if there is one.
    mlir::Attribute filler;
    if (value.hasArrayFiller()) {
      filler = tryEmitAbstractForMemory(value.getArrayFiller(),
                                        arrayTy->getElementType());
      if (!filler)
        return {};
    }

    // Emit initializer elements.
    SmallVector<mlir::TypedAttr, 16> elts;
    if (filler && builder.isNullValue(filler))
      elts.reserve(numInitElts + 1);
    else
      elts.reserve(numElements);

    mlir::Type commonElementType;
    for (unsigned i = 0; i < numInitElts; ++i) {
      auto c = tryEmitPrivateForMemory(value.getArrayInitializedElt(i),
                                       arrayTy->getElementType());
      if (!c)
        return {};

      assert(mlir::isa<mlir::TypedAttr>(c) &&
             "This should always be a TypedAttr.");
      auto cTyped = mlir::cast<mlir::TypedAttr>(c);

      if (i == 0)
        commonElementType = cTyped.getType();
      else if (cTyped.getType() != commonElementType)
        commonElementType = {};
      auto typedC = llvm::dyn_cast<mlir::TypedAttr>(c);
      if (!typedC)
        llvm_unreachable("this should always be typed");
      elts.push_back(typedC);
    }

    auto desired = CGM.getTypes().ConvertType(destType);

    auto typedFiller = llvm::dyn_cast_or_null<mlir::TypedAttr>(filler);
    if (filler && !typedFiller)
      llvm_unreachable("this should always be typed");

    return buildArrayConstant(CGM, desired, commonElementType, numElements,
                              elts, typedFiller);
  }
  case APValue::Vector: {
    const QualType elementType =
        destType->castAs<VectorType>()->getElementType();
    unsigned numElements = value.getVectorLength();
    SmallVector<mlir::Attribute, 16> elts;
    elts.reserve(numElements);
    for (unsigned i = 0; i < numElements; ++i) {
      auto c = tryEmitPrivateForMemory(value.getVectorElt(i), elementType);
      if (!c)
        return {};
      elts.push_back(c);
    }
    auto desired =
        mlir::cast<mlir::cir::VectorType>(CGM.getTypes().ConvertType(destType));
    return mlir::cir::ConstVectorAttr::get(
        desired, mlir::ArrayAttr::get(CGM.getBuilder().getContext(), elts));
  }
  case APValue::MemberPointer: {
    assert(!MissingFeatures::cxxABI());

    const ValueDecl *memberDecl = value.getMemberPointerDecl();
    assert(!value.isMemberPointerToDerivedMember() && "NYI");

    if (const auto *memberFuncDecl = dyn_cast<CXXMethodDecl>(memberDecl))
      assert(0 && "not implemented");

    auto cirTy = mlir::cast<mlir::cir::DataMemberType>(
        CGM.getTypes().ConvertType(destType));

    const auto *fieldDecl = cast<FieldDecl>(memberDecl);
    return builder.getDataMemberAttr(cirTy, fieldDecl->getFieldIndex());
  }
  case APValue::LValue:
    return ConstantLValueEmitter(*this, value, destType).tryEmit();
  case APValue::Struct:
  case APValue::Union:
    return ConstStructBuilder::buildStruct(*this, value, destType);
  case APValue::FixedPoint:
  case APValue::ComplexInt:
  case APValue::ComplexFloat:
  case APValue::AddrLabelDiff:
    assert(0 && "not implemented");
  }
  llvm_unreachable("Unknown APValue kind");
}

mlir::Value CIRGenModule::buildNullConstant(QualType t, mlir::Location loc) {
  if (t->getAs<PointerType>()) {
    return builder.getNullPtr(getTypes().convertTypeForMem(t), loc);
  }

  if (getTypes().isZeroInitializable(t))
    return builder.getNullValue(getTypes().convertTypeForMem(t), loc);

  if (const ConstantArrayType *cat =
          getASTContext().getAsConstantArrayType(t)) {
    llvm_unreachable("NYI");
  }

  if (const RecordType *rt = t->getAs<RecordType>())
    llvm_unreachable("NYI");

  assert(t->isMemberDataPointerType() &&
         "Should only see pointers to data members here!");

  llvm_unreachable("NYI");
  return {};
}

mlir::Value CIRGenModule::buildMemberPointerConstant(const UnaryOperator *e) {
  assert(!MissingFeatures::cxxABI());

  auto loc = getLoc(e->getSourceRange());

  const auto *decl = cast<DeclRefExpr>(e->getSubExpr())->getDecl();

  // A member function pointer.
  if (const auto *methodDecl = dyn_cast<CXXMethodDecl>(decl)) {
    auto ty = mlir::cast<mlir::cir::MethodType>(getCIRType(e->getType()));
    if (methodDecl->isVirtual())
      return builder.create<mlir::cir::ConstantOp>(
          loc, ty, getCXXABI().buildVirtualMethodAttr(ty, methodDecl));

    auto methodFuncOp = GetAddrOfFunction(methodDecl);
    return builder.create<mlir::cir::ConstantOp>(
        loc, ty, builder.getMethodAttr(ty, methodFuncOp));
  }

  auto ty = mlir::cast<mlir::cir::DataMemberType>(getCIRType(e->getType()));

  // Otherwise, a member data pointer.
  const auto *fieldDecl = cast<FieldDecl>(decl);
  return builder.create<mlir::cir::ConstantOp>(
      loc, ty, builder.getDataMemberAttr(ty, fieldDecl->getFieldIndex()));
}

mlir::Attribute ConstantEmitter::emitAbstract(const Expr *e,
                                              QualType destType) {
  auto state = pushAbstract();
  auto c = mlir::cast<mlir::Attribute>(tryEmitPrivate(e, destType));
  c = validateAndPopAbstract(c, state);
  if (!c) {
    llvm_unreachable("NYI");
  }
  return c;
}

mlir::Attribute ConstantEmitter::emitAbstract(SourceLocation loc,
                                              const APValue &value,
                                              QualType destType) {
  auto state = pushAbstract();
  auto c = tryEmitPrivate(value, destType);
  c = validateAndPopAbstract(c, state);
  if (!c) {
    CGM.Error(loc,
              "internal error: could not emit constant value \"abstractly\"");
    llvm_unreachable("NYI");
  }
  return c;
}

mlir::Attribute ConstantEmitter::emitNullForMemory(mlir::Location loc,
                                                   CIRGenModule &cgm,
                                                   QualType t) {
  auto cstOp = dyn_cast<mlir::cir::ConstantOp>(
      cgm.buildNullConstant(t, loc).getDefiningOp());
  assert(cstOp && "expected cir.const op");
  return emitForMemory(cgm, cstOp.getValue(), t);
}
