#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Interfaces for AllocaOp
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> cir::AllocaOp::getPromotableSlots() {
  return {MemorySlot{getResult(), getAllocaType()}};
}

Value cir::AllocaOp::getDefaultValue(const MemorySlot &slot,
                                      RewriterBase &rewriter) {
  return rewriter.create<cir::UndefOp>(getLoc(), slot.elemType);
}

void cir::AllocaOp::handleBlockArgument(const MemorySlot &slot,
                                         BlockArgument argument,
                                         RewriterBase &rewriter) {}

void cir::AllocaOp::handlePromotionComplete(const MemorySlot &slot,
                                            Value defaultValue,
                                            RewriterBase &rewriter) {
  if (defaultValue && defaultValue.use_empty())
    rewriter.eraseOp(defaultValue.getDefiningOp());
  rewriter.eraseOp(*this);
}

//===----------------------------------------------------------------------===//
// Interfaces for LoadOp
//===----------------------------------------------------------------------===//

bool cir::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

bool cir::LoadOp::storesTo(const MemorySlot &slot) { return false; }

Value cir::LoadOp::getStored(const MemorySlot &slot, RewriterBase &rewriter,
                              Value reachingDef, const DataLayout &dataLayout) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool cir::LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getResult().getType() == slot.elemType;
}

DeletionKind cir::LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition,
    const DataLayout &dataLayout) {
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Interfaces for StoreOp
//===----------------------------------------------------------------------===//

bool cir::StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool cir::StoreOp::storesTo(const MemorySlot &slot) {
  return getAddr() == slot.ptr;  
}

Value cir::StoreOp::getStored(const MemorySlot &slot, RewriterBase &rewriter,
                              Value reachingDef, const DataLayout &dataLayout) {
  return getValue();
}

bool cir::StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr &&
         slot.elemType == getValue().getType();
}

DeletionKind cir::StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition,
    const DataLayout &dataLayout) {
  return DeletionKind::Delete;
}


//===----------------------------------------------------------------------===//
// Interfaces for CopyOp
//===----------------------------------------------------------------------===//


bool cir::CopyOp::loadsFrom(const MemorySlot &slot) {
  return getSrc() == slot.ptr;
}

bool cir::CopyOp::storesTo(const MemorySlot &slot) {
  return getDst() == slot.ptr;
}

Value cir::CopyOp::getStored(const MemorySlot &slot, RewriterBase &rewriter,
                             Value reachingDef,
                             const DataLayout &dataLayout) {
  return rewriter.create<cir::LoadOp>(getLoc(), slot.elemType, getSrc());
}


DeletionKind cir::CopyOp::removeBlockingUses(const MemorySlot &slot,
                         const SmallPtrSetImpl<OpOperand *> &blockingUses,
                         RewriterBase &rewriter, Value reachingDefinition,
                         const DataLayout &dataLayout) {
  if (loadsFrom(slot))
    rewriter.create<cir::StoreOp>(getLoc(), reachingDefinition, getDst(), 
                                  false, mlir::cir::MemOrderAttr());
  return DeletionKind::Delete;
}

bool cir::CopyOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  
  if (getDst() == getSrc())
    return false;

  return getLength() == dataLayout.getTypeSize(slot.elemType);
}

/// Conditions the deletion of the operation to the removal of all its uses.
static bool forwardToUsers(Operation *op,
                           SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

bool cir::GetMemberOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  // GetMemberOp can be removed as long as it is a no-op and its users can be removed.
  if (getIndex() != 0)
    return false; 
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind cir::GetMemberOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
}

