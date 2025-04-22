//===-- CIRGenBuilder.cpp - CIRBuilder implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CIRGenBuilder.h"

using namespace clang::CIRGen;

mlir::Value CIRGenBuilderTy::maybeBuildArrayDecay(mlir::Location loc,
                                                  mlir::Value arrayPtr,
                                                  mlir::Type eltTy) {
  auto arrayPtrTy = ::mlir::dyn_cast<cir::PointerType>(arrayPtr.getType());
  assert(arrayPtrTy && "expected pointer type");
  auto arrayTy = ::mlir::dyn_cast<cir::ArrayType>(arrayPtrTy.getPointee());

  if (arrayTy) {
    auto addrSpace = ::mlir::cast_if_present<cir::AddressSpaceAttr>(
        arrayPtrTy.getAddrSpace());
    cir::PointerType flatPtrTy = getPointerTo(arrayTy.getEltType(), addrSpace);
    return create<cir::CastOp>(loc, flatPtrTy, cir::CastKind::array_to_ptrdecay,
                               arrayPtr);
  }

  assert(arrayPtrTy.getPointee() == eltTy &&
         "flat pointee type must match original array element type");
  return arrayPtr;
}

mlir::Value CIRGenBuilderTy::getArrayElement(mlir::Location arrayLocBegin,
                                             mlir::Location arrayLocEnd,
                                             mlir::Value arrayPtr,
                                             mlir::Type eltTy, mlir::Value idx,
                                             bool shouldDecay) {
  mlir::Value basePtr = arrayPtr;
  if (shouldDecay)
    basePtr = maybeBuildArrayDecay(arrayLocBegin, arrayPtr, eltTy);
  mlir::Type flatPtrTy = basePtr.getType();
  return create<cir::PtrStrideOp>(arrayLocEnd, flatPtrTy, basePtr, idx);
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APSInt intVal) {
  bool isSigned = intVal.isSigned();
  auto width = intVal.getBitWidth();
  cir::IntType t = isSigned ? getSIntNTy(width) : getUIntNTy(width);
  return getConstInt(loc, t,
                     isSigned ? intVal.getSExtValue() : intVal.getZExtValue());
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc,
                                             llvm::APInt intVal) {
  auto width = intVal.getBitWidth();
  cir::IntType t = getUIntNTy(width);
  return getConstInt(loc, t, intVal.getZExtValue());
}

cir::ConstantOp CIRGenBuilderTy::getConstInt(mlir::Location loc, mlir::Type t,
                                             uint64_t C) {
  auto intTy = mlir::dyn_cast<cir::IntType>(t);
  assert(intTy && "expected cir::IntType");
  return create<cir::ConstantOp>(loc, intTy, cir::IntAttr::get(t, C));
}


struct SubTypeVisitor {
  using onSubType = std::function<void(mlir::Type, int64_t, uint64_t)>;

  SubTypeVisitor(cir::CIRDataLayout layout, onSubType fun) 
    : layout_(layout)
    , onSubType_(fun)  {}
 
  void visit(mlir::Type t, int64_t offset) const {
    if (!offset)
      return;
    if (auto array = mlir::dyn_cast<cir::ArrayType>(t)) {
      visitArrayType(array, offset);
    } else if (auto record = mlir::dyn_cast<cir::RecordType>(t)) {
      visitRecordType(record, offset);
    } else {
      llvm_unreachable("unexpected type");
    }
  }

private:

  void visitArrayType(cir::ArrayType ar, int64_t offset) const {
    int64_t eltSize = layout_.getTypeAllocSize(ar.getEltType());  
    int64_t divRet = offset / eltSize;    
    if (divRet < 0)
      divRet -= 1; // make sure offset is positive    

    int64_t newOffset = offset - (divRet * eltSize);
    onSubType_(ar.getEltType(), newOffset, divRet);
    visit(ar.getEltType(), newOffset);
  }

  void visitRecordType(cir::RecordType rec, int64_t offset) const {
    auto elts = rec.getMembers();
    int64_t pos = 0;
    size_t i = 0;
    for (; i < elts.size(); ++i) {
      int64_t eltSize =
          (int64_t)layout_.getTypeAllocSize(elts[i]).getFixedValue();
      unsigned alignMask = layout_.getABITypeAlign(elts[i]).value() - 1;
      if (rec.getPacked())
        alignMask = 0;
      pos = (pos + alignMask) & ~alignMask;
      assert(offset >= 0);
      if (offset < pos + eltSize) {
        offset -= pos;
        break;
      }
      pos += eltSize;
    }
    onSubType_(elts[i], offset, i);    
    visit(elts[i], offset);
  }

private:
  cir::CIRDataLayout layout_;
  onSubType onSubType_;
};


void CIRGenBuilderTy::computeGlobalViewIndicesFromFlatOffset(
    int64_t Offset, mlir::Type Ty, cir::CIRDataLayout Layout,
    llvm::SmallVectorImpl<int64_t> &Indices) {

  auto pushIndex = [&](mlir::Type t, int64_t offset, uint64_t index) {
    Indices.push_back(index);
  };

  SubTypeVisitor v(Layout, pushIndex);
  v.visit(Ty, Offset);
}

bool CIRGenBuilderTy::isOffsetInUnion(cir::CIRDataLayout layout, 
      mlir::Type typ, int64_t offset) {

  auto isUnion = [](mlir::Type t) {
    if (auto rec = mlir::dyn_cast<cir::RecordType>(t))
      return rec.isUnion();
    return false;
  };

  if (isUnion(typ))
    return true;    
  
  bool result = false;
  
  auto check = [&](mlir::Type t, uint64_t offset, uint64_t index) {
    result = result || isUnion(t);
  };

  SubTypeVisitor v(layout, check);
  v.visit(typ, offset);

  return result;
} 

uint64_t CIRGenBuilderTy::computeOffsetFromGlobalViewIndices(
    const cir::CIRDataLayout &layout, mlir::Type typ,
    llvm::ArrayRef<int64_t> indexes) {

  int64_t offset = 0;
  for (int64_t idx : indexes) {
    if (auto sTy = dyn_cast<cir::RecordType>(typ)) {
      offset += sTy.getElementOffset(layout.layout, idx);
      assert(idx < (int64_t)sTy.getMembers().size());
      typ = sTy.getMembers()[idx];
    } else if (auto arTy = dyn_cast<cir::ArrayType>(typ)) {
      typ = arTy.getEltType();
      offset += layout.getTypeAllocSize(typ) * idx;
    } else {
      llvm_unreachable("NYI");
    }
  }

  return offset;
}

// This can't be defined in Address.h because that file is included by
// CIRGenBuilder.h
Address Address::withElementType(CIRGenBuilderTy &builder,
                                 mlir::Type ElemTy) const {
  if (!hasOffset())
    return Address(builder.createPtrBitcast(getBasePointer(), ElemTy), ElemTy,
                   getAlignment(), getPointerAuthInfo(), /*Offset=*/nullptr,
                   isKnownNonNull());
  return Address(builder.createPtrBitcast(getPointer(), ElemTy), ElemTy,
                 getAlignment(), isKnownNonNull());
}
