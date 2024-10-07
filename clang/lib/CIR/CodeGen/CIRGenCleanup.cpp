//===--- CIRGenCleanup.cpp - Bookkeeping and code emission for cleanups ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code dealing with the IR generation for cleanups
// and related information.
//
// A "cleanup" is a piece of code which needs to be executed whenever
// control transfers out of a particular scope.  This can be
// conditionalized to occur only on exceptional control flow, only on
// normal control flow, or both.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SaveAndRestore.h"

#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// CIRGenFunction cleanup related
//===----------------------------------------------------------------------===//

/// Build a unconditional branch to the lexical scope cleanup block
/// or with the labeled blocked if already solved.
///
/// Track on scope basis, goto's we need to fix later.
mlir::cir::BrOp CIRGenFunction::buildBranchThroughCleanup(mlir::Location loc,
                                                          JumpDest dest) {
  // Remove this once we go for making sure unreachable code is
  // well modeled (or not).
  assert(builder.getInsertionBlock() && "not yet implemented");
  assert(!MissingFeatures::ehStack());

  // Insert a branch: to the cleanup block (unsolved) or to the already
  // materialized label. Keep track of unsolved goto's.
  return builder.create<BrOp>(loc, dest.isValid() ? dest.getBlock()
                                                  : ReturnBlock().getBlock());
}

/// Emits all the code to cause the given temporary to be cleaned up.
void CIRGenFunction::buildCXXTemporary(const CXXTemporary *temporary,
                                       QualType tempType, Address ptr) {
  pushDestroy(NormalAndEHCleanup, ptr, tempType, destroyCXXObject,
              /*useEHCleanup*/ true);
}

Address CIRGenFunction::createCleanupActiveFlag() {
  mlir::Location loc = currSrcLoc ? *currSrcLoc : builder.getUnknownLoc();

  // Create a variable to decide whether the cleanup needs to be run.
  // FIXME: set the insertion point for the alloca to be at the entry
  // basic block of the previous scope, not the entry block of the function.
  Address active = CreateTempAllocaWithoutCast(
      builder.getBoolTy(), CharUnits::One(), loc, "cleanup.cond");
  mlir::Value falseVal, trueVal;
  {
    // Place true/false flags close to their allocas.
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(active.getPointer());
    falseVal = builder.getFalse(loc);
    trueVal = builder.getTrue(loc);
  }

  // Initialize it to false at a site that's guaranteed to be run
  // before each evaluation.
  setBeforeOutermostConditional(falseVal, active);

  // Initialize it to true at the current location.
  builder.createStore(loc, trueVal, active);
  return active;
}

DominatingValue<RValue>::saved_type
DominatingValue<RValue>::saved_type::save(CIRGenFunction &cgf, RValue rv) {
  if (rv.isScalar()) {
    mlir::Value val = rv.getScalarVal();
    return saved_type(DominatingCIRValue::save(cgf, val),
                      DominatingCIRValue::needsSaving(val) ? ScalarAddress
                                                           : ScalarLiteral);
  }

  if (rv.isComplex()) {
    llvm_unreachable("complex NYI");
  }

  llvm_unreachable("aggregate NYI");
}

/// Given a saved r-value produced by SaveRValue, perform the code
/// necessary to restore it to usability at the current insertion
/// point.
RValue DominatingValue<RValue>::saved_type::restore(CIRGenFunction &cgf) {
  switch (K) {
  case ScalarLiteral:
  case ScalarAddress:
    return RValue::get(DominatingCIRValue::restore(cgf, Vals.first));
  case AggregateLiteral:
  case AggregateAddress:
    return RValue::getAggregate(
        DominatingValue<Address>::restore(cgf, AggregateAddr));
  case ComplexAddress: {
    llvm_unreachable("NYI");
  }
  }

  llvm_unreachable("bad saved r-value kind");
}

static bool isUsedAsEhCleanup(EHScopeStack &ehStack,
                              EHScopeStack::stable_iterator cleanup) {
  // If we needed an EH block for any reason, that counts.
  if (ehStack.find(cleanup)->hasEHBranches())
    return true;

  // Check whether any enclosed cleanups were needed.
  for (EHScopeStack::stable_iterator i = ehStack.getInnermostEHScope();
       i != cleanup;) {
    assert(cleanup.strictlyEncloses(i));

    EHScope &scope = *ehStack.find(i);
    if (scope.hasEHBranches())
      return true;

    i = scope.getEnclosingEHScope();
  }

  return false;
}

enum ForActivationT { ForActivation, ForDeactivation };

/// The given cleanup block is changing activation state.  Configure a
/// cleanup variable if necessary.
///
/// It would be good if we had some way of determining if there were
/// extra uses *after* the change-over point.
static void setupCleanupBlockActivation(CIRGenFunction &cgf,
                                        EHScopeStack::stable_iterator c,
                                        ForActivationT kind,
                                        mlir::Operation *dominatingIP) {
  EHCleanupScope &scope = cast<EHCleanupScope>(*cgf.EHStack.find(c));

  // We always need the flag if we're activating the cleanup in a
  // conditional context, because we have to assume that the current
  // location doesn't necessarily dominate the cleanup's code.
  bool isActivatedInConditional =
      (kind == ForActivation && cgf.isInConditionalBranch());

  bool needFlag = false;

  // Calculate whether the cleanup was used:

  //   - as a normal cleanup
  if (scope.isNormalCleanup()) {
    scope.setTestFlagInNormalCleanup();
    needFlag = true;
  }

  //  - as an EH cleanup
  if (scope.isEHCleanup() &&
      (isActivatedInConditional || isUsedAsEhCleanup(cgf.EHStack, c))) {
    scope.setTestFlagInEHCleanup();
    needFlag = true;
  }

  // If it hasn't yet been used as either, we're done.
  if (!needFlag)
    return;

  Address var = scope.getActiveFlag();
  if (!var.isValid()) {
    llvm_unreachable("NYI");
  }

  auto builder = cgf.getBuilder();
  mlir::Location loc = var.getPointer().getLoc();
  mlir::Value trueOrFalse =
      kind == ForActivation ? builder.getTrue(loc) : builder.getFalse(loc);
  cgf.getBuilder().createStore(loc, trueOrFalse, var);
}

/// Deactive a cleanup that was created in an active state.
void CIRGenFunction::DeactivateCleanupBlock(EHScopeStack::stable_iterator c,
                                            mlir::Operation *dominatingIP) {
  assert(c != EHStack.stable_end() && "deactivating bottom of stack?");
  EHCleanupScope &scope = cast<EHCleanupScope>(*EHStack.find(c));
  assert(scope.isActive() && "double deactivation");

  // If it's the top of the stack, just pop it, but do so only if it belongs
  // to the current RunCleanupsScope.
  if (c == EHStack.stable_begin() &&
      CurrentCleanupScopeDepth.strictlyEncloses(c)) {
    // Per comment below, checking EHAsynch is not really necessary
    // it's there to assure zero-impact w/o EHAsynch option
    if (!scope.isNormalCleanup() && getLangOpts().EHAsynch) {
      llvm_unreachable("NYI");
    } else {
      // From LLVM: If it's a normal cleanup, we need to pretend that the
      // fallthrough is unreachable.
      // CIR remarks: LLVM uses an empty insertion point to signal behavior
      // change to other codegen paths (triggered by PopCleanupBlock).
      // CIRGen doesn't do that yet, but let's mimic just in case.
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.clearInsertionPoint();
      PopCleanupBlock();
    }
    return;
  }

  // Otherwise, follow the general case.
  setupCleanupBlockActivation(*this, c, ForDeactivation, dominatingIP);
  scope.setActive(false);
}

void CIRGenFunction::initFullExprCleanupWithFlag(Address activeFlag) {
  // Set that as the active flag in the cleanup.
  EHCleanupScope &cleanup = cast<EHCleanupScope>(*EHStack.begin());
  assert(!cleanup.hasActiveFlag() && "cleanup already has active flag?");
  cleanup.setActiveFlag(activeFlag);

  if (cleanup.isNormalCleanup())
    cleanup.setTestFlagInNormalCleanup();
  if (cleanup.isEHCleanup())
    cleanup.setTestFlagInEHCleanup();
}

/// We don't need a normal entry block for the given cleanup.
/// Optimistic fixup branches can cause these blocks to come into
/// existence anyway;  if so, destroy it.
///
/// The validity of this transformation is very much specific to the
/// exact ways in which we form branches to cleanup entries.
static void destroyOptimisticNormalEntry(CIRGenFunction &cgf,
                                         EHCleanupScope &scope) {
  auto *entry = scope.getNormalBlock();
  if (!entry)
    return;

  llvm_unreachable("NYI");
}

static void buildCleanup(CIRGenFunction &cgf, EHScopeStack::Cleanup *fn,
                         EHScopeStack::Cleanup::Flags flags,
                         Address activeFlag) {
  auto emitCleanup = [&]() {
    // Ask the cleanup to emit itself.
    assert(cgf.HaveInsertPoint() && "expected insertion point");
    fn->Emit(cgf, flags);
    assert(cgf.HaveInsertPoint() && "cleanup ended with no insertion point?");
  };

  // If there's an active flag, load it and skip the cleanup if it's
  // false.
  cir::CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc =
      cgf.currSrcLoc ? *cgf.currSrcLoc : builder.getUnknownLoc();

  if (activeFlag.isValid()) {
    mlir::Value isActive = builder.createLoad(loc, activeFlag);
    builder.create<mlir::cir::IfOp>(loc, isActive, false,
                                    [&](mlir::OpBuilder &b, mlir::Location) {
                                      emitCleanup();
                                      builder.createYield(loc);
                                    });
  } else {
    emitCleanup();
  }
  // No need to emit continuation block because CIR uses a cir.if.
}

/// Pops a cleanup block. If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CIRGenFunction::PopCleanupBlock(bool fallthroughIsBranchThrough) {
  assert(!EHStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*EHStack.begin()) && "top not a cleanup!");
  EHCleanupScope &scope = cast<EHCleanupScope>(*EHStack.begin());
  assert(scope.getFixupDepth() <= EHStack.getNumBranchFixups());

  // Remember activation information.
  bool isActive = scope.isActive();
  Address normalActiveFlag = scope.shouldTestFlagInNormalCleanup()
                                 ? scope.getActiveFlag()
                                 : Address::invalid();
  Address ehActiveFlag = scope.shouldTestFlagInEHCleanup()
                             ? scope.getActiveFlag()
                             : Address::invalid();

  // Check whether we need an EH cleanup. This is only true if we've
  // generated a lazy EH cleanup block.
  auto *ehEntry = scope.getCachedEHDispatchBlock();
  assert(scope.hasEHBranches() == (ehEntry != nullptr));
  bool requiresEhCleanup = (ehEntry != nullptr);
  EHScopeStack::stable_iterator ehParent = scope.getEnclosingEHScope();

  // Check the three conditions which might require a normal cleanup:

  // - whether there are branch fix-ups through this cleanup
  unsigned fixupDepth = scope.getFixupDepth();
  bool hasFixups = EHStack.getNumBranchFixups() != fixupDepth;

  // - whether there are branch-throughs or branch-afters
  bool hasExistingBranches = scope.hasBranches();

  // - whether there's a fallthrough
  auto *fallthroughSource = builder.getInsertionBlock();
  bool hasFallthrough = (fallthroughSource != nullptr && isActive);

  // Branch-through fall-throughs leave the insertion point set to the
  // end of the last cleanup, which points to the current scope.  The
  // rest of CIR gen doesn't need to worry about this; it only happens
  // during the execution of PopCleanupBlocks().
  bool hasTerminator = fallthroughSource &&
                       fallthroughSource->mightHaveTerminator() &&
                       fallthroughSource->getTerminator();
  bool hasPrebranchedFallthrough =
      hasTerminator &&
      !isa<mlir::cir::YieldOp>(fallthroughSource->getTerminator());

  // If this is a normal cleanup, then having a prebranched
  // fallthrough implies that the fallthrough source unconditionally
  // jumps here.
  assert(!scope.isNormalCleanup() || !hasPrebranchedFallthrough ||
         (scope.getNormalBlock() &&
          fallthroughSource->getTerminator()->getSuccessor(0) ==
              scope.getNormalBlock()));

  bool requiresNormalCleanup = false;
  if (scope.isNormalCleanup() &&
      (hasFixups || hasExistingBranches || hasFallthrough)) {
    requiresNormalCleanup = true;
  }

  // If we have a prebranched fallthrough into an inactive normal
  // cleanup, rewrite it so that it leads to the appropriate place.
  if (scope.isNormalCleanup() && hasPrebranchedFallthrough && !isActive) {
    llvm_unreachable("NYI");
  }

  // If we don't need the cleanup at all, we're done.
  if (!requiresNormalCleanup && !requiresEhCleanup) {
    destroyOptimisticNormalEntry(*this, scope);
    EHStack.popCleanup(); // safe because there are no fixups
    assert(EHStack.getNumBranchFixups() == 0 || EHStack.hasNormalCleanups());
    return;
  }

  // Copy the cleanup emission data out.  This uses either a stack
  // array or malloc'd memory, depending on the size, which is
  // behavior that SmallVector would provide, if we could use it
  // here. Unfortunately, if you ask for a SmallVector<char>, the
  // alignment isn't sufficient.
  auto *cleanupSource = reinterpret_cast<char *>(scope.getCleanupBuffer());
  alignas(EHScopeStack::ScopeStackAlignment) char
      cleanupBufferStack[8 * sizeof(void *)];
  std::unique_ptr<char[]> cleanupBufferHeap;
  size_t cleanupSize = scope.getCleanupSize();
  EHScopeStack::Cleanup *fn;

  if (cleanupSize <= sizeof(cleanupBufferStack)) {
    memcpy(cleanupBufferStack, cleanupSource, cleanupSize);
    fn = reinterpret_cast<EHScopeStack::Cleanup *>(cleanupBufferStack);
  } else {
    cleanupBufferHeap.reset(new char[cleanupSize]);
    memcpy(cleanupBufferHeap.get(), cleanupSource, cleanupSize);
    fn = reinterpret_cast<EHScopeStack::Cleanup *>(cleanupBufferHeap.get());
  }

  EHScopeStack::Cleanup::Flags cleanupFlags;
  if (scope.isNormalCleanup())
    cleanupFlags.setIsNormalCleanupKind();
  if (scope.isEHCleanup())
    cleanupFlags.setIsEHCleanupKind();

  // Under -EHa, invoke seh.scope.end() to mark scope end before dtor
  bool isEHa = getLangOpts().EHAsynch && !scope.isLifetimeMarker();
  // const EHPersonality &Personality = EHPersonality::get(*this);
  if (!requiresNormalCleanup) {
    // Mark CPP scope end for passed-by-value Arg temp
    //   per Windows ABI which is "normally" Cleanup in callee
    if (isEHa && isInvokeDest()) {
      // If we are deactivating a normal cleanup then we don't have a
      // fallthrough. Restore original IP to emit CPP scope ends in the correct
      // block.
      llvm_unreachable("NYI");
    }
    destroyOptimisticNormalEntry(*this, scope);
    scope.markEmitted();
    EHStack.popCleanup();
  } else {
    // If we have a fallthrough and no other need for the cleanup,
    // emit it directly.
    if (hasFallthrough && !hasPrebranchedFallthrough && !hasFixups &&
        !hasExistingBranches) {

      // mark SEH scope end for fall-through flow
      if (isEHa) {
        llvm_unreachable("NYI");
      }

      destroyOptimisticNormalEntry(*this, scope);
      EHStack.popCleanup();
      scope.markEmitted();
      buildCleanup(*this, fn, cleanupFlags, normalActiveFlag);

      // Otherwise, the best approach is to thread everything through
      // the cleanup block and then try to clean up after ourselves.
    } else {
      llvm_unreachable("NYI");
    }
  }

  assert(EHStack.hasNormalCleanups() || EHStack.getNumBranchFixups() == 0);

  // Emit the EH cleanup if required.
  if (requiresEhCleanup) {
    mlir::cir::TryOp tryOp =
        ehEntry->getParentOp()->getParentOfType<mlir::cir::TryOp>();
    auto *nextAction = getEHDispatchBlock(ehParent, tryOp);
    (void)nextAction;

    // Push a terminate scope or cleanupendpad scope around the potentially
    // throwing cleanups. For funclet EH personalities, the cleanupendpad models
    // program termination when cleanups throw.
    bool pushedTerminate = false;
    SaveAndRestore restoreCurrentFuncletPad(CurrentFuncletPad);
    mlir::Operation *cpi = nullptr;

    const EHPersonality &personality = EHPersonality::get(*this);
    if (personality.usesFuncletPads()) {
      llvm_unreachable("NYI");
    }

    // Non-MSVC personalities need to terminate when an EH cleanup throws.
    if (!personality.isMSVCPersonality()) {
      EHStack.pushTerminate();
      pushedTerminate = true;
    } else if (isEHa && isInvokeDest()) {
      llvm_unreachable("NYI");
    }

    // We only actually emit the cleanup code if the cleanup is either
    // active or was used before it was deactivated.
    if (ehActiveFlag.isValid() || isActive) {
      cleanupFlags.setIsForEHCleanup();
      mlir::OpBuilder::InsertionGuard guard(builder);

      auto yield = cast<YieldOp>(ehEntry->getTerminator());
      builder.setInsertionPoint(yield);
      buildCleanup(*this, fn, cleanupFlags, ehActiveFlag);
    }

    if (cpi)
      llvm_unreachable("NYI");
    else {
      // In LLVM traditional codegen, here's where it branches off to
      // nextAction. CIR does not have a flat layout at this point, so
      // instead patch all the landing pads that need to run this cleanup
      // as well.
      mlir::Block *currBlock = ehEntry;
      while (currBlock && cleanupsToPatch.contains(currBlock)) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block *blockToPatch = cleanupsToPatch[currBlock];
        auto currYield = cast<YieldOp>(blockToPatch->getTerminator());
        builder.setInsertionPoint(currYield);

        // If nextAction is an EH resume block, also update all try locations
        // for these "to-patch" blocks with the appropriate resume content.
        if (nextAction == ehResumeBlock) {
          if (auto tryToPatch = currYield->getParentOp()
                                    ->getParentOfType<mlir::cir::TryOp>()) {
            mlir::Block *resumeBlockToPatch =
                tryToPatch.getCatchUnwindEntryBlock();
            buildEHResumeBlock(/*isCleanup=*/true, resumeBlockToPatch,
                               tryToPatch.getLoc());
          }
        }

        buildCleanup(*this, fn, cleanupFlags, ehActiveFlag);
        currBlock = blockToPatch;
      }

      // The nextAction is yet to be populated, register that this
      // cleanup should also incorporate any cleanup from nextAction
      // when available.
      cleanupsToPatch[nextAction] = ehEntry;
    }

    // Leave the terminate scope.
    if (pushedTerminate)
      EHStack.popTerminate();

    // FIXME(cir): LLVM traditional codegen tries to simplify some of the
    // codegen here. Once we are further down with EH support revisit whether we
    // need to this during lowering.
    assert(!MissingFeatures::simplifyCleanupEntry());
  }
}

/// Pops cleanup blocks until the given savepoint is reached.
void CIRGenFunction::PopCleanupBlocks(
    EHScopeStack::stable_iterator old,
    std::initializer_list<mlir::Value *> valuesToReload) {
  assert(old.isValid());

  bool hadBranches = false;
  while (EHStack.stable_begin() != old) {
    EHCleanupScope &scope = cast<EHCleanupScope>(*EHStack.begin());
    hadBranches |= scope.hasBranches();

    // As long as Old strictly encloses the scope's enclosing normal
    // cleanup, we're going to emit another normal cleanup which
    // fallthrough can propagate through.
    bool fallThroughIsBranchThrough =
        old.strictlyEncloses(scope.getEnclosingNormalCleanup());

    PopCleanupBlock(fallThroughIsBranchThrough);
  }

  // If we didn't have any branches, the insertion point before cleanups must
  // dominate the current insertion point and we don't need to reload any
  // values.
  if (!hadBranches)
    return;

  llvm_unreachable("NYI");
}

/// Pops cleanup blocks until the given savepoint is reached, then add the
/// cleanups from the given savepoint in the lifetime-extended cleanups stack.
void CIRGenFunction::PopCleanupBlocks(
    EHScopeStack::stable_iterator old, size_t oldLifetimeExtendedSize,
    std::initializer_list<mlir::Value *> valuesToReload) {
  PopCleanupBlocks(old, valuesToReload);

  // Move our deferred cleanups onto the EH stack.
  for (size_t i = oldLifetimeExtendedSize,
              e = LifetimeExtendedCleanupStack.size();
       i != e;
       /**/) {
    // Alignment should be guaranteed by the vptrs in the individual cleanups.
    assert((i % alignof(LifetimeExtendedCleanupHeader) == 0) &&
           "misaligned cleanup stack entry");

    LifetimeExtendedCleanupHeader &header =
        reinterpret_cast<LifetimeExtendedCleanupHeader &>(
            LifetimeExtendedCleanupStack[i]);
    i += sizeof(header);

    EHStack.pushCopyOfCleanup(
        header.getKind(), &LifetimeExtendedCleanupStack[i], header.getSize());
    i += header.getSize();

    if (header.isConditional()) {
      Address activeFlag =
          reinterpret_cast<Address &>(LifetimeExtendedCleanupStack[i]);
      initFullExprCleanupWithFlag(activeFlag);
      i += sizeof(activeFlag);
    }
  }
  LifetimeExtendedCleanupStack.resize(oldLifetimeExtendedSize);
}

//===----------------------------------------------------------------------===//
// EHScopeStack
//===----------------------------------------------------------------------===//

void EHScopeStack::Cleanup::anchor() {}

/// Push an entry of the given size onto this protected-scope stack.
char *EHScopeStack::allocate(size_t size) {
  size = llvm::alignTo(size, ScopeStackAlignment);
  if (!StartOfBuffer) {
    unsigned capacity = 1024;
    while (capacity < size)
      capacity *= 2;
    StartOfBuffer = new char[capacity];
    StartOfData = EndOfBuffer = StartOfBuffer + capacity;
  } else if (static_cast<size_t>(StartOfData - StartOfBuffer) < size) {
    unsigned currentCapacity = EndOfBuffer - StartOfBuffer;
    unsigned usedCapacity = currentCapacity - (StartOfData - StartOfBuffer);

    unsigned newCapacity = currentCapacity;
    do {
      newCapacity *= 2;
    } while (newCapacity < usedCapacity + size);

    char *newStartOfBuffer = new char[newCapacity];
    char *newEndOfBuffer = newStartOfBuffer + newCapacity;
    char *newStartOfData = newEndOfBuffer - usedCapacity;
    memcpy(newStartOfData, StartOfData, usedCapacity);
    delete[] StartOfBuffer;
    StartOfBuffer = newStartOfBuffer;
    EndOfBuffer = newEndOfBuffer;
    StartOfData = newStartOfData;
  }

  assert(StartOfBuffer + size <= StartOfData);
  StartOfData -= size;
  return StartOfData;
}

void *EHScopeStack::pushCleanup(CleanupKind kind, size_t size) {
  char *buffer = allocate(EHCleanupScope::getSizeForCleanupSize(size));
  bool isNormalCleanup = kind & NormalCleanup;
  bool isEhCleanup = kind & EHCleanup;
  bool isLifetimeMarker = kind & LifetimeMarker;

  // Per C++ [except.terminate], it is implementation-defined whether none,
  // some, or all cleanups are called before std::terminate. Thus, when
  // terminate is the current EH scope, we may skip adding any EH cleanup
  // scopes.
  if (InnermostEHScope != stable_end() &&
      find(InnermostEHScope)->getKind() == EHScope::Terminate)
    isEhCleanup = false;

  EHCleanupScope *scope = new (buffer)
      EHCleanupScope(isNormalCleanup, isEhCleanup, size, BranchFixups.size(),
                     InnermostNormalCleanup, InnermostEHScope);
  if (isNormalCleanup)
    InnermostNormalCleanup = stable_begin();
  if (isEhCleanup)
    InnermostEHScope = stable_begin();
  if (isLifetimeMarker)
    llvm_unreachable("NYI");

  // With Windows -EHa, Invoke llvm.seh.scope.begin() for EHCleanup
  if (CGF->getLangOpts().EHAsynch && isEhCleanup && !isLifetimeMarker &&
      CGF->getTarget().getCXXABI().isMicrosoft())
    llvm_unreachable("NYI");

  return scope->getCleanupBuffer();
}

void EHScopeStack::popCleanup() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHCleanupScope>(*begin()));
  EHCleanupScope &cleanup = cast<EHCleanupScope>(*begin());
  InnermostNormalCleanup = cleanup.getEnclosingNormalCleanup();
  InnermostEHScope = cleanup.getEnclosingEHScope();
  deallocate(cleanup.getAllocatedSize());

  // Destroy the cleanup.
  cleanup.Destroy();

  // Check whether we can shrink the branch-fixups stack.
  if (!BranchFixups.empty()) {
    // If we no longer have any normal cleanups, all the fixups are
    // complete.
    if (!hasNormalCleanups())
      BranchFixups.clear();

    // Otherwise we can still trim out unnecessary nulls.
    else
      popNullFixups();
  }
}

void EHScopeStack::deallocate(size_t size) {
  StartOfData += llvm::alignTo(size, ScopeStackAlignment);
}

/// Remove any 'null' fixups on the stack.  However, we can't pop more
/// fixups than the fixup depth on the innermost normal cleanup, or
/// else fixups that we try to add to that cleanup will end up in the
/// wrong place.  We *could* try to shrink fixup depths, but that's
/// actually a lot of work for little benefit.
void EHScopeStack::popNullFixups() {
  // We expect this to only be called when there's still an innermost
  // normal cleanup;  otherwise there really shouldn't be any fixups.
  llvm_unreachable("NYI");
}

bool EHScopeStack::requiresLandingPad() const {
  for (stable_iterator si = getInnermostEHScope(); si != stable_end();) {
    // Skip lifetime markers.
    if (auto *cleanup = dyn_cast<EHCleanupScope>(&*find(si)))
      if (cleanup->isLifetimeMarker()) {
        si = cleanup->getEnclosingEHScope();
        continue;
      }
    return true;
  }

  return false;
}

EHCatchScope *EHScopeStack::pushCatch(unsigned numHandlers) {
  char *buffer = allocate(EHCatchScope::getSizeForNumHandlers(numHandlers));
  EHCatchScope *scope =
      new (buffer) EHCatchScope(numHandlers, InnermostEHScope);
  InnermostEHScope = stable_begin();
  return scope;
}

void EHScopeStack::pushTerminate() {
  char *buffer = allocate(EHTerminateScope::getSize());
  new (buffer) EHTerminateScope(InnermostEHScope);
  InnermostEHScope = stable_begin();
}