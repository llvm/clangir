//===--- ConstantInitBuilder.cpp - Global initializer builder -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines out-of-line routines for building initializers for
// global variables, in particular the kind of globals that are implicitly
// introduced by various language ABIs.
//
//===----------------------------------------------------------------------===//

#include "ConstantInitBuilder.h"
#include "CIRGenModule.h"

using namespace clang;
using namespace cir;

ConstantInitBuilderBase::ConstantInitBuilderBase(CIRGenModule &CGM)
    : CGM(CGM), builder(CGM.getBuilder()) {}

mlir::Type ConstantInitFuture::getType() const {
  assert(Data && "dereferencing null future");
  if (Data.is<mlir::Attribute>()) {
    auto attr = Data.get<mlir::Attribute>().dyn_cast<mlir::TypedAttr>();
    assert(attr && "expected typed attribute");
    return attr.getType();
  } else {
    llvm_unreachable("Only sypport typed attributes here");
  }
}

void ConstantInitFuture::abandon() {
  assert(Data && "abandoning null future");
  if (auto builder = Data.dyn_cast<ConstantInitBuilderBase *>()) {
    builder->abandon(0);
  }
  Data = nullptr;
}

void ConstantInitFuture::installInGlobal(mlir::cir::GlobalOp GV) {
  assert(Data && "installing null future");
  if (Data.is<mlir::Attribute>()) {
    GV.setInitialValueAttr(Data.get<mlir::Attribute>());
  } else {
    llvm_unreachable("NYI");
    // auto &builder = *Data.get<ConstantInitBuilderBase *>();
    // assert(builder.Buffer.size() == 1);
    // builder.setGlobalInitializer(GV, builder.Buffer[0]);
    // builder.Buffer.clear();
    // Data = nullptr;
  }
}

ConstantInitFuture
ConstantInitBuilderBase::createFuture(mlir::Attribute initializer) {
  assert(Buffer.empty() && "buffer not current empty");
  Buffer.push_back(initializer);
  return ConstantInitFuture(this);
}

// Only used in this file.
inline ConstantInitFuture::ConstantInitFuture(ConstantInitBuilderBase *builder)
    : Data(builder) {
  assert(!builder->Frozen);
  assert(builder->Buffer.size() == 1);
  assert(builder->Buffer[0] != nullptr);
}

mlir::cir::GlobalOp ConstantInitBuilderBase::createGlobal(
    mlir::Attribute initializer, const llvm::Twine &name, CharUnits alignment,
    bool constant, mlir::cir::GlobalLinkageKind linkage,
    unsigned addressSpace) {
  llvm_unreachable("NYI");
  // auto GV =
  //     new llvm::GlobalVariable(CGM.getModule(), initializer->getType(),
  //                              constant, linkage, initializer, name,
  //                              /*insert before*/ nullptr,
  //                              llvm::GlobalValue::NotThreadLocal,
  //                              addressSpace);
  // GV->setAlignment(alignment.getAsAlign());
  // resolveSelfReferences(GV);
  // return GV;
}

void ConstantInitBuilderBase::setGlobalInitializer(
    mlir::cir::GlobalOp GV, mlir::Attribute initializer) {
  // GV->setInitializer(initializer);

  // if (!SelfReferences.empty())
  //   resolveSelfReferences(GV);
  llvm_unreachable("NYI");
}

void ConstantInitBuilderBase::resolveSelfReferences(mlir::cir::GlobalOp GV) {
  llvm_unreachable("NYI");
  // for (auto &entry : SelfReferences) {
  //   mlir::Attribute resolvedReference =
  //       llvm::ConstantExpr::getInBoundsGetElementPtr(GV->getValueType(), GV,
  //                                                    entry.Indices);
  //   auto dummy = entry.Dummy;
  //   dummy->replaceAllUsesWith(resolvedReference);
  //   dummy->eraseFromParent();
  // }
  // SelfReferences.clear();
}

void ConstantInitBuilderBase::abandon(size_t newEnd) {
  llvm_unreachable("NYI");
  // // Remove all the entries we've added.
  // Buffer.erase(Buffer.begin() + newEnd, Buffer.end());

  // // If we're abandoning all the way to the beginning, destroy
  // // all the self-references, because we might not get another
  // // opportunity.
  // if (newEnd == 0) {
  //   for (auto &entry : SelfReferences) {
  //     auto dummy = entry.Dummy;
  //     dummy->replaceAllUsesWith(llvm::PoisonValue::get(dummy->getType()));
  //     dummy->eraseFromParent();
  //   }
  //   SelfReferences.clear();
  // }
}

void ConstantAggregateBuilderBase::addSize(CharUnits size) {
  add(Builder.CGM.getSize(size));
}

mlir::Attribute
ConstantAggregateBuilderBase::getRelativeOffset(mlir::IntegerType offsetType,
                                                mlir::Attribute target) {
  return getRelativeOffsetToPosition(offsetType, target,
                                     Builder.Buffer.size() - Begin);
}

mlir::Attribute ConstantAggregateBuilderBase::getRelativeOffsetToPosition(
    mlir::IntegerType offsetType, mlir::Attribute target, size_t position) {
  llvm_unreachable("NYI");
  // // Compute the address of the relative-address slot.
  // auto base = getAddrOfPosition(offsetType, position);

  // // Subtract.
  // base = llvm::ConstantExpr::getPtrToInt(base, Builder.CGM.IntPtrTy);
  // target = llvm::ConstantExpr::getPtrToInt(target, Builder.CGM.IntPtrTy);
  // mlir::Attribute offset = llvm::ConstantExpr::getSub(target, base);

  // // Truncate to the relative-address type if necessary.
  // if (Builder.CGM.IntPtrTy != offsetType) {
  //   offset = llvm::ConstantExpr::getTrunc(offset, offsetType);
  // }

  // return offset;
}

mlir::Attribute
ConstantAggregateBuilderBase::getAddrOfPosition(mlir::Type type,
                                                size_t position) {
  llvm_unreachable("NYI");
  // // Make a global variable.  We will replace this with a GEP to this
  // // position after installing the initializer.
  // auto dummy = new llvm::GlobalVariable(Builder.CGM.getModule(), type, true,
  //                                       llvm::GlobalVariable::PrivateLinkage,
  //                                       nullptr, "");
  // Builder.SelfReferences.emplace_back(dummy);
  // auto &entry = Builder.SelfReferences.back();
  // (void)getGEPIndicesTo(entry.Indices, position + Begin);
  // return dummy;
}

mlir::Attribute
ConstantAggregateBuilderBase::getAddrOfCurrentPosition(mlir::Type type) {
  llvm_unreachable("NYI");
  // // Make a global variable.  We will replace this with a GEP to this
  // // position after installing the initializer.
  // auto dummy = new llvm::GlobalVariable(Builder.CGM.getModule(), type, true,
  //                                       llvm::GlobalVariable::PrivateLinkage,
  //                                       nullptr, "");
  // Builder.SelfReferences.emplace_back(dummy);
  // auto &entry = Builder.SelfReferences.back();
  // (void)getGEPIndicesToCurrentPosition(entry.Indices);
  // return dummy;
}

void ConstantAggregateBuilderBase::getGEPIndicesTo(
    llvm::SmallVectorImpl<mlir::Attribute> &indices, size_t position) const {
  llvm_unreachable("NYI");
  // // Recurse on the parent builder if present.
  // if (Parent) {
  //   Parent->getGEPIndicesTo(indices, Begin);

  //   // Otherwise, add an index to drill into the first level of pointer.
  // } else {
  //   assert(indices.empty());
  //   indices.push_back(llvm::ConstantInt::get(Builder.CGM.Int32Ty, 0));
  // }

  // assert(position >= Begin);
  // // We have to use i32 here because struct GEPs demand i32 indices.
  // // It's rather unlikely to matter in practice.
  // indices.push_back(
  //     llvm::ConstantInt::get(Builder.CGM.Int32Ty, position - Begin));
}

ConstantAggregateBuilderBase::PlaceholderPosition
ConstantAggregateBuilderBase::addPlaceholderWithSize(mlir::Type type) {
  llvm_unreachable("NYI");
  // // Bring the offset up to the last field.
  // CharUnits offset = getNextOffsetFromGlobal();

  // // Create the placeholder.
  // auto position = addPlaceholder();

  // // Advance the offset past that field.
  // auto &layout = Builder.CGM.getDataLayout();
  // if (!Packed)
  //   offset =
  //       offset.alignTo(CharUnits::fromQuantity(layout.getABITypeAlign(type)));
  // offset += CharUnits::fromQuantity(layout.getTypeStoreSize(type));

  // CachedOffsetEnd = Builder.Buffer.size();
  // CachedOffsetFromGlobal = offset;

  // return position;
}

CharUnits
ConstantAggregateBuilderBase::getOffsetFromGlobalTo(size_t end) const {
  size_t cacheEnd = CachedOffsetEnd;
  assert(cacheEnd <= end);

  // Fast path: if the cache is valid, just use it.
  if (cacheEnd == end) {
    return CachedOffsetFromGlobal;
  }

  // If the cached range ends before the index at which the current
  // aggregate starts, recurse for the parent.
  CharUnits offset;
  if (cacheEnd < Begin) {
    assert(cacheEnd == 0);
    assert(Parent && "Begin != 0 for root builder");
    cacheEnd = Begin;
    offset = Parent->getOffsetFromGlobalTo(Begin);
  } else {
    offset = CachedOffsetFromGlobal;
  }

  // Perform simple layout on the elements in cacheEnd..<end.
  if (cacheEnd != end) {
    llvm_unreachable("NYI");
    // auto &layout = Builder.CGM.getDataLayout();
    // do {
    //   mlir::Attribute element = Builder.Buffer[cacheEnd];
    //   assert(element != nullptr &&
    //          "cannot compute offset when a placeholder is present");
    //   mlir::Type elementType = element->getType();
    //   if (!Packed)
    //     offset = offset.alignTo(
    //         CharUnits::fromQuantity(layout.getABITypeAlign(elementType)));
    //   offset +=
    //   CharUnits::fromQuantity(layout.getTypeStoreSize(elementType));
    // } while (++cacheEnd != end);
  }

  // Cache and return.
  CachedOffsetEnd = cacheEnd;
  CachedOffsetFromGlobal = offset;
  return offset;
}

mlir::Attribute ConstantAggregateBuilderBase::finishArray(mlir::Type eltTy) {
  llvm_unreachable("NYI");
  // markFinished();

  // auto &buffer = getBuffer();
  // assert((Begin < buffer.size() || (Begin == buffer.size() && eltTy)) &&
  //        "didn't add any array elements without element type");
  // auto elts = llvm::ArrayRef(buffer).slice(Begin);
  // if (!eltTy)
  //   eltTy = elts[0]->getType();
  // auto type = llvm::ArrayType::get(eltTy, elts.size());
  // auto constant = llvm::ConstantArray::get(type, elts);
  // buffer.erase(buffer.begin() + Begin, buffer.end());
  // return constant;
}

mlir::Attribute
ConstantAggregateBuilderBase::finishStruct(mlir::cir::StructType ty) {
  llvm_unreachable("NYI");
  // markFinished();

  // auto &buffer = getBuffer();
  // auto elts = llvm::ArrayRef(buffer).slice(Begin);

  // if (ty == nullptr && elts.empty())
  //   ty = mlir::cir::StructType::get(Builder.CGM.getLLVMContext(), {},
  //   Packed);

  // mlir::Attribute constant;
  // if (ty) {
  //   assert(ty->isPacked() == Packed);
  //   constant = llvm::ConstantStruct::get(ty, elts);
  // } else {
  //   constant = llvm::ConstantStruct::getAnon(elts, Packed);
  // }

  // buffer.erase(buffer.begin() + Begin, buffer.end());
  // return constant;
}
