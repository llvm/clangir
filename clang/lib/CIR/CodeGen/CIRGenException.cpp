//===--- CIRGenException.cpp - Emit CIR Code for C++ exceptions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ exception related code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace cir;
using namespace clang;

const EHPersonality EHPersonality::GNU_C = {"__gcc_personality_v0", nullptr};
const EHPersonality EHPersonality::GNU_C_SJLJ = {"__gcc_personality_sj0",
                                                 nullptr};
const EHPersonality EHPersonality::GNU_C_SEH = {"__gcc_personality_seh0",
                                                nullptr};
const EHPersonality EHPersonality::NeXT_ObjC = {"__objc_personality_v0",
                                                nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus = {"__gxx_personality_v0",
                                                    nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SJLJ = {
    "__gxx_personality_sj0", nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SEH = {
    "__gxx_personality_seh0", nullptr};
const EHPersonality EHPersonality::GNU_ObjC = {"__gnu_objc_personality_v0",
                                               "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SJLJ = {
    "__gnu_objc_personality_sj0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SEH = {
    "__gnu_objc_personality_seh0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjCXX = {
    "__gnustep_objcxx_personality_v0", nullptr};
const EHPersonality EHPersonality::GNUstep_ObjC = {
    "__gnustep_objc_personality_v0", nullptr};
const EHPersonality EHPersonality::MSVC_except_handler = {"_except_handler3",
                                                          nullptr};
const EHPersonality EHPersonality::MSVC_C_specific_handler = {
    "__C_specific_handler", nullptr};
const EHPersonality EHPersonality::MSVC_CxxFrameHandler3 = {
    "__CxxFrameHandler3", nullptr};
const EHPersonality EHPersonality::GNU_Wasm_CPlusPlus = {
    "__gxx_wasm_personality_v0", nullptr};
const EHPersonality EHPersonality::XL_CPlusPlus = {"__xlcxx_personality_v1",
                                                   nullptr};

static const EHPersonality &getCPersonality(const TargetInfo &target,
                                            const LangOptions &l) {
  const llvm::Triple &t = target.getTriple();
  if (t.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (l.hasSjLjExceptions())
    return EHPersonality::GNU_C_SJLJ;
  if (l.hasDWARFExceptions())
    return EHPersonality::GNU_C;
  if (l.hasSEHExceptions())
    return EHPersonality::GNU_C_SEH;
  return EHPersonality::GNU_C;
}

static const EHPersonality &getObjCPersonality(const TargetInfo &target,
                                               const LangOptions &l) {
  const llvm::Triple &t = target.getTriple();
  if (t.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (l.ObjCRuntime.getKind()) {
  case ObjCRuntime::FragileMacOSX:
    return getCPersonality(target, l);
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return EHPersonality::NeXT_ObjC;
  case ObjCRuntime::GNUstep:
    if (l.ObjCRuntime.getVersion() >= VersionTuple(1, 7))
      return EHPersonality::GNUstep_ObjC;
    [[fallthrough]];
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    if (l.hasSjLjExceptions())
      return EHPersonality::GNU_ObjC_SJLJ;
    if (l.hasSEHExceptions())
      return EHPersonality::GNU_ObjC_SEH;
    return EHPersonality::GNU_ObjC;
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getCXXPersonality(const TargetInfo &target,
                                              const LangOptions &l) {
  const llvm::Triple &t = target.getTriple();
  if (t.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (t.isOSAIX())
    return EHPersonality::XL_CPlusPlus;
  if (l.hasSjLjExceptions())
    return EHPersonality::GNU_CPlusPlus_SJLJ;
  if (l.hasDWARFExceptions())
    return EHPersonality::GNU_CPlusPlus;
  if (l.hasSEHExceptions())
    return EHPersonality::GNU_CPlusPlus_SEH;
  if (l.hasWasmExceptions())
    return EHPersonality::GNU_Wasm_CPlusPlus;
  return EHPersonality::GNU_CPlusPlus;
}

/// Determines the personality function to use when both C++
/// and Objective-C exceptions are being caught.
static const EHPersonality &getObjCXXPersonality(const TargetInfo &target,
                                                 const LangOptions &l) {
  if (target.getTriple().isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (l.ObjCRuntime.getKind()) {
  // In the fragile ABI, just use C++ exception handling and hope
  // they're not doing crazy exception mixing.
  case ObjCRuntime::FragileMacOSX:
    return getCXXPersonality(target, l);

  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return getObjCPersonality(target, l);

  case ObjCRuntime::GNUstep:
    return EHPersonality::GNU_ObjCXX;

  // The GCC runtime's personality function inherently doesn't support
  // mixed EH.  Use the ObjC personality just to avoid returning null.
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    return getObjCPersonality(target, l);
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getSEHPersonalityMSVC(const llvm::Triple &t) {
  if (t.getArch() == llvm::Triple::x86)
    return EHPersonality::MSVC_except_handler;
  return EHPersonality::MSVC_C_specific_handler;
}

const EHPersonality &EHPersonality::get(CIRGenModule &cgm,
                                        const FunctionDecl *fd) {
  const llvm::Triple &t = cgm.getTarget().getTriple();
  const LangOptions &l = cgm.getLangOpts();
  const TargetInfo &target = cgm.getTarget();

  // Functions using SEH get an SEH personality.
  if (fd && fd->usesSEHTry())
    return getSEHPersonalityMSVC(t);

  if (l.ObjC)
    return l.CPlusPlus ? getObjCXXPersonality(target, l)
                       : getObjCPersonality(target, l);
  return l.CPlusPlus ? getCXXPersonality(target, l)
                     : getCPersonality(target, l);
}

const EHPersonality &EHPersonality::get(CIRGenFunction &cgf) {
  const auto *fd = cgf.CurCodeDecl;
  // For outlined finallys and filters, use the SEH personality in case they
  // contain more SEH. This mostly only affects finallys. Filters could
  // hypothetically use gnu statement expressions to sneak in nested SEH.
  fd = fd ? fd : cgf.CurSEHParent.getDecl();
  return get(cgf.cgm, dyn_cast_or_null<FunctionDecl>(fd));
}

void CIRGenFunction::buildCXXThrowExpr(const CXXThrowExpr *e) {
  if (const Expr *subExpr = e->getSubExpr()) {
    QualType throwType = subExpr->getType();
    if (throwType->isObjCObjectPointerType()) {
      llvm_unreachable("NYI");
    } else {
      cgm.getCXXABI().buildThrow(*this, e);
    }
  } else {
    cgm.getCXXABI().buildRethrow(*this, /*isNoReturn=*/true);
  }

  // In LLVM codegen the expression emitters expect to leave this
  // path by starting a new basic block. We do not need that in CIR.
}

namespace {
/// A cleanup to free the exception object if its initialization
/// throws.
struct FreeException final : EHScopeStack::Cleanup {
  mlir::Value exn;
  FreeException(mlir::Value exn) : exn(exn) {}
  void Emit(CIRGenFunction &cgf, Flags flags) override {
    llvm_unreachable("call to cxa_free or equivalent op NYI");
  }
};
} // end anonymous namespace

// Emits an exception expression into the given location.  This
// differs from buildAnyExprToMem only in that, if a final copy-ctor
// call is required, an exception within that copy ctor causes
// std::terminate to be invoked.
void CIRGenFunction::buildAnyExprToExn(const Expr *e, Address addr) {
  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  pushFullExprCleanup<FreeException>(EHCleanup, addr.getPointer());
  EHScopeStack::stable_iterator cleanup = EHStack.stable_begin();

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  auto ty = convertTypeForMem(e->getType());
  Address typedAddr = addr.withElementType(ty);

  // From LLVM's codegen:
  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  buildAnyExprToMem(e, typedAddr, e->getType().getQualifiers(),
                    /*IsInit*/ true);

  // Deactivate the cleanup block.
  auto *op = typedAddr.getPointer().getDefiningOp();
  assert(op &&
         "expected valid Operation *, block arguments are not meaningful here");
  DeactivateCleanupBlock(cleanup, op);
}

void CIRGenFunction::buildEHResumeBlock(bool isCleanup,
                                        mlir::Block *ehResumeBlock,
                                        mlir::Location loc) {
  auto ip = getBuilder().saveInsertionPoint();
  getBuilder().setInsertionPointToStart(ehResumeBlock);

  const EHPersonality &personality = EHPersonality::get(*this);

  // This can always be a call
  // because we necessarily didn't
  // find anything on the EH stack
  // which needs our help.
  const char *rethrowName = personality.CatchallRethrowFn;
  if (rethrowName != nullptr && !isCleanup) {
    // FIXME(cir): upon testcase
    // this should just add the
    // 'rethrow' attribute to
    // mlir::cir::ResumeOp below.
    llvm_unreachable("NYI");
  }

  getBuilder().create<mlir::cir::ResumeOp>(loc, mlir::Value{}, mlir::Value{});
  getBuilder().restoreInsertionPoint(ip);
}

mlir::Block *CIRGenFunction::getEHResumeBlock(bool isCleanup,
                                              mlir::cir::TryOp tryOp) {

  if (ehResumeBlock)
    return ehResumeBlock;
  // Setup unwind.
  assert(tryOp && "expected available cir.try");
  ehResumeBlock = tryOp.getCatchUnwindEntryBlock();
  if (!ehResumeBlock->empty())
    return ehResumeBlock;

  buildEHResumeBlock(isCleanup, ehResumeBlock, tryOp.getLoc());
  return ehResumeBlock;
}

mlir::LogicalResult CIRGenFunction::buildCXXTryStmt(const CXXTryStmt &tryStmt) {
  auto loc = getLoc(tryStmt.getSourceRange());
  mlir::OpBuilder::InsertPoint scopeIP;

  // Create a scope to hold try local storage for catch params.
  [[maybe_unused]] auto scopeOp = builder.create<mlir::cir::ScopeOp>(
      loc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        scopeIP = getBuilder().saveInsertionPoint();
      });

  auto r = mlir::success();
  {
    mlir::OpBuilder::InsertionGuard guard(getBuilder());
    getBuilder().restoreInsertionPoint(scopeIP);
    r = buildCXXTryStmtUnderScope(tryStmt);
    getBuilder().create<mlir::cir::YieldOp>(loc);
  }
  return r;
}

mlir::LogicalResult
CIRGenFunction::buildCXXTryStmtUnderScope(const CXXTryStmt &S) {
  const llvm::Triple &t = getTarget().getTriple();
  // If we encounter a try statement on in an OpenMP target region offloaded to
  // a GPU, we treat it as a basic block.
  const bool isTargetDevice =
      (cgm.getLangOpts().OpenMPIsTargetDevice && (t.isNVPTX() || t.isAMDGCN()));
  assert(!isTargetDevice && "NYI");

  auto hasCatchAll = [&]() {
    if (!S.getNumHandlers())
      return false;
    unsigned lastHandler = S.getNumHandlers() - 1;
    return S.getHandler(lastHandler)->getExceptionDecl() == nullptr;
  };

  auto numHandlers = S.getNumHandlers();
  auto tryLoc = getLoc(S.getBeginLoc());

  mlir::OpBuilder::InsertPoint beginInsertTryBody;

  // Create the scope to represent only the C/C++ `try {}` part. However,
  // don't populate right away. Reserve some space to store the exception
  // info but don't emit the bulk right away, for now only make sure the
  // scope returns the exception information.
  auto tryOp = builder.create<mlir::cir::TryOp>(
      tryLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        beginInsertTryBody = getBuilder().saveInsertionPoint();
      },
      // Don't emit the code right away for catch clauses, for
      // now create the regions and consume the try scope result.
      // Note that clauses are later populated in
      // CIRGenFunction::buildLandingPad.
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::OperationState &result) {
        mlir::OpBuilder::InsertionGuard guard(b);
        auto numRegionsToCreate = numHandlers;
        if (!hasCatchAll())
          numRegionsToCreate++;
        // Once for each handler + (catch_all or unwind).
        for (int i = 0, e = numRegionsToCreate; i != e; ++i) {
          auto *r = result.addRegion();
          builder.createBlock(r);
        }
      });

  // Finally emit the body for try/catch.
  auto emitTryCatchBody = [&]() -> mlir::LogicalResult {
    auto loc = tryOp.getLoc();
    mlir::OpBuilder::InsertionGuard guard(getBuilder());
    getBuilder().restoreInsertionPoint(beginInsertTryBody);
    CIRGenFunction::LexicalScope tryScope{*this, loc,
                                          getBuilder().getInsertionBlock()};

    {
      tryScope.setAsTry(tryOp);
      // Attach the basic blocks for the catch regions.
      enterCXXTryStmt(S, tryOp);
      // Emit the body for the `try {}` part.
      {
        CIRGenFunction::LexicalScope tryBodyScope{
            *this, loc, getBuilder().getInsertionBlock()};
        if (buildStmt(S.getTryBlock(), /*useCurrentScope=*/true).failed())
          return mlir::failure();
      }
    }

    {
      // Emit catch clauses.
      exitCXXTryStmt(S);
    }

    return mlir::success();
  };

  return emitTryCatchBody();
}

/// Emit the structure of the dispatch block for the given catch scope.
/// It is an invariant that the dispatch block already exists.
static void buildCatchDispatchBlock(CIRGenFunction &cgf,
                                    EHCatchScope &catchScope,
                                    mlir::cir::TryOp tryOp) {
  if (EHPersonality::get(cgf).isWasmPersonality())
    llvm_unreachable("NYI");
  if (EHPersonality::get(cgf).usesFuncletPads())
    llvm_unreachable("NYI");

  auto *dispatchBlock = catchScope.getCachedEHDispatchBlock();
  assert(dispatchBlock);

  // If there's only a single catch-all, getEHDispatchBlock returned
  // that catch-all as the dispatch block.
  if (catchScope.getNumHandlers() == 1 &&
      catchScope.getHandler(0).isCatchAll()) {
    // assert(dispatchBlock == catchScope.getHandler(0).Block);
    return;
  }

  // In traditional LLVM codegen, the right handler is selected (with
  // calls to eh_typeid_for) and the selector value is loaded. After that,
  // blocks get connected for later codegen. In CIR, these are all
  // implicit behaviors of cir.catch - not a lot of work to do.
  //
  // Test against each of the exception types we claim to catch.
  for (unsigned i = 0, e = catchScope.getNumHandlers();; ++i) {
    assert(i < e && "ran off end of handlers!");
    const EHCatchScope::Handler &handler = catchScope.getHandler(i);

    auto typeValue = handler.Type.RTTI;
    assert(handler.Type.Flags == 0 && "catch handler flags not supported");
    assert(typeValue && "fell into catch-all case!");
    // Check for address space mismatch: if (typeValue->getType() !=
    // argTy)
    assert(!MissingFeatures::addressSpace());

    bool nextIsEnd = false;
    // If this is the last handler, we're at the end, and the next
    // block is the block for the enclosing EH scope. Make sure to call
    // getEHDispatchBlock for caching it.
    if (i + 1 == e) {
      (void)cgf.getEHDispatchBlock(catchScope.getEnclosingEHScope(), tryOp);
      nextIsEnd = true;

      // If the next handler is a catch-all, we're at the end, and the
      // next block is that handler.
    } else if (catchScope.getHandler(i + 1).isCatchAll()) {
      // Block already created when creating catch regions, just mark this
      // is the end.
      nextIsEnd = true;
    }

    // If the next handler is a catch-all, we're completely done.
    if (nextIsEnd)
      return;
  }
}

void CIRGenFunction::enterCXXTryStmt(const CXXTryStmt &s,
                                     mlir::cir::TryOp tryOp,
                                     bool isFnTryBlock) {
  unsigned numHandlers = s.getNumHandlers();
  EHCatchScope *catchScope = EHStack.pushCatch(numHandlers);
  for (unsigned i = 0; i != numHandlers; ++i) {
    const CXXCatchStmt *c = s.getHandler(i);

    mlir::Block *handler = &tryOp.getCatchRegions()[i].getBlocks().front();
    if (c->getExceptionDecl()) {
      // FIXME: Dropping the reference type on the type into makes it
      // impossible to correctly implement catch-by-reference
      // semantics for pointers.  Unfortunately, this is what all
      // existing compilers do, and it's not clear that the standard
      // personality routine is capable of doing this right.  See C++ DR 388 :
      // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#388
      Qualifiers caughtTypeQuals;
      QualType caughtType = cgm.getASTContext().getUnqualifiedArrayType(
          c->getCaughtType().getNonReferenceType(), caughtTypeQuals);

      CatchTypeInfo typeInfo{nullptr, 0};
      if (caughtType->isObjCObjectPointerType())
        llvm_unreachable("NYI");
      else
        typeInfo = cgm.getCXXABI().getAddrOfCXXCatchHandlerType(
            getLoc(s.getSourceRange()), caughtType, c->getCaughtType());
      catchScope->setHandler(i, typeInfo, handler);
    } else {
      // No exception decl indicates '...', a catch-all.
      catchScope->setHandler(i, cgm.getCXXABI().getCatchAllTypeInfo(), handler);
      // Under async exceptions, catch(...) need to catch HW exception too
      // Mark scope with SehTryBegin as a SEH __try scope
      if (getLangOpts().EHAsynch)
        llvm_unreachable("NYI");
    }
  }
}

void CIRGenFunction::exitCXXTryStmt(const CXXTryStmt &s, bool isFnTryBlock) {
  unsigned numHandlers = s.getNumHandlers();
  EHCatchScope &catchScope = cast<EHCatchScope>(*EHStack.begin());
  assert(catchScope.getNumHandlers() == numHandlers);
  mlir::cir::TryOp tryOp = currLexScope->getTry();

  // If the catch was not required, bail out now.
  if (!catchScope.hasEHBranches()) {
    catchScope.clearHandlerBlocks();
    EHStack.popCatch();
    // Drop all basic block from all catch regions.
    SmallVector<mlir::Block *> eraseBlocks;
    for (mlir::Region &r : tryOp.getCatchRegions()) {
      if (r.empty())
        continue;
      for (mlir::Block &b : r.getBlocks())
        eraseBlocks.push_back(&b);
    }
    for (mlir::Block *b : eraseBlocks)
      b->erase();
    tryOp.setCatchTypesAttr({});
    return;
  }

  // Emit the structure of the EH dispatch for this catch.
  buildCatchDispatchBlock(*this, catchScope, tryOp);

  // Copy the handler blocks off before we pop the EH stack.  Emitting
  // the handlers might scribble on this memory.
  SmallVector<EHCatchScope::Handler, 8> handlers(
      catchScope.begin(), catchScope.begin() + numHandlers);

  EHStack.popCatch();

  // Determine if we need an implicit rethrow for all these catch handlers;
  // see the comment below.
  bool doImplicitRethrow = false;
  if (isFnTryBlock)
    doImplicitRethrow = isa<CXXDestructorDecl>(CurCodeDecl) ||
                        isa<CXXConstructorDecl>(CurCodeDecl);

  // Wasm uses Windows-style EH instructions, but merges all catch clauses into
  // one big catchpad. So we save the old funclet pad here before we traverse
  // each catch handler.
  SaveAndRestore restoreCurrentFuncletPad(CurrentFuncletPad);
  mlir::Block *wasmCatchStartBlock = nullptr;
  if (EHPersonality::get(*this).isWasmPersonality()) {
    llvm_unreachable("NYI");
  }

  bool hasCatchAll = false;
  for (unsigned i = numHandlers; i != 0; --i) {
    hasCatchAll |= handlers[i - 1].isCatchAll();
    mlir::Block *catchBlock = handlers[i - 1].Block;
    mlir::OpBuilder::InsertionGuard guard(getBuilder());
    getBuilder().setInsertionPointToStart(catchBlock);

    // Catch the exception if this isn't a catch-all.
    const CXXCatchStmt *c = s.getHandler(i - 1);

    // Enter a cleanup scope, including the catch variable and the
    // end-catch.
    RunCleanupsScope catchScope(*this);

    // Initialize the catch variable and set up the cleanups.
    SaveAndRestore restoreCurrentFuncletPad(CurrentFuncletPad);
    cgm.getCXXABI().emitBeginCatch(*this, c);

    // Emit the PGO counter increment.
    assert(!MissingFeatures::incrementProfileCounter());

    // Perform the body of the catch.
    (void)buildStmt(c->getHandlerBlock(), /*useCurrentScope=*/true);

    // [except.handle]p11:
    //   The currently handled exception is rethrown if control
    //   reaches the end of a handler of the function-try-block of a
    //   constructor or destructor.

    // It is important that we only do this on fallthrough and not on
    // return.  Note that it's illegal to put a return in a
    // constructor function-try-block's catch handler (p14), so this
    // really only applies to destructors.
    if (doImplicitRethrow && HaveInsertPoint()) {
      llvm_unreachable("NYI");
    }

    // Fall out through the catch cleanups.
    catchScope.ForceCleanup();
  }

  // Because in wasm we merge all catch clauses into one big catchpad, in case
  // none of the types in catch handlers matches after we test against each   of
  // them, we should unwind to the next EH enclosing scope. We generate a   call
  // to rethrow function here to do that.
  if (EHPersonality::get(*this).isWasmPersonality() && !hasCatchAll) {
    assert(wasmCatchStartBlock);
    // Navigate for the "rethrow" block we created in emitWasmCatchPadBlock().
    // Wasm uses landingpad-style conditional branches to compare selectors, so
    // we follow the false destination for each of the cond branches to reach
    // the rethrow block.
    llvm_unreachable("NYI");
  }

  assert(!MissingFeatures::incrementProfileCounter());
}

/// Check whether this is a non-EH scope, i.e. a scope which doesn't
/// affect exception handling.  Currently, the only non-EH scopes are
/// normal-only cleanup scopes.
[[maybe_unused]] static bool isNonEHScope(const EHScope &s) {
  switch (s.getKind()) {
  case EHScope::Cleanup:
    return !cast<EHCleanupScope>(s).isEHCleanup();
  case EHScope::Filter:
  case EHScope::Catch:
  case EHScope::Terminate:
    return false;
  }

  llvm_unreachable("Invalid EHScope Kind!");
}

mlir::Operation *CIRGenFunction::buildLandingPad(mlir::cir::TryOp tryOp) {
  assert(EHStack.requiresLandingPad());
  assert(!cgm.getLangOpts().IgnoreExceptions &&
         "LandingPad should not be emitted when -fignore-exceptions are in "
         "effect.");
  EHScope &innermostEHScope = *EHStack.find(EHStack.getInnermostEHScope());
  switch (innermostEHScope.getKind()) {
  case EHScope::Terminate:
    return getTerminateLandingPad();

  case EHScope::Catch:
  case EHScope::Cleanup:
  case EHScope::Filter:
    // CIR does not cache landing pads.
    break;
  }

  // If there's an existing TryOp, it means we got a `cir.try` scope
  // that leads to this "landing pad" creation site. Otherwise, exceptions
  // are enabled but a throwing function is called anyways (common pattern
  // with function local static initializers).
  mlir::ArrayAttr catches = tryOp.getCatchTypesAttr();
  if (!catches || catches.empty()) {
    // Save the current CIR generation state.
    mlir::OpBuilder::InsertionGuard guard(builder);
    assert(!MissingFeatures::generateDebugInfo() && "NYI");

    // Traditional LLVM codegen creates the lpad basic block, extract
    // values, landing pad instructions, etc.

    // Accumulate all the handlers in scope.
    bool hasCatchAll = false;
    bool hasCleanup = false;
    bool hasFilter = false;
    SmallVector<mlir::Value, 4> filterTypes;
    llvm::SmallPtrSet<mlir::Attribute, 4> catchTypes;
    SmallVector<mlir::Attribute, 4> clauses;

    for (EHScopeStack::iterator i = EHStack.begin(), e = EHStack.end(); i != e;
         ++i) {

      switch (i->getKind()) {
      case EHScope::Cleanup:
        // If we have a cleanup, remember that.
        hasCleanup = (hasCleanup || cast<EHCleanupScope>(*i).isEHCleanup());
        continue;

      case EHScope::Filter: {
        llvm_unreachable("NYI");
      }

      case EHScope::Terminate:
        // Terminate scopes are basically catch-alls.
        // assert(!hasCatchAll);
        // hasCatchAll = true;
        // goto done;
        llvm_unreachable("NYI");

      case EHScope::Catch:
        break;
      }

      EHCatchScope &catchScope = cast<EHCatchScope>(*i);
      for (unsigned hi = 0, he = catchScope.getNumHandlers(); hi != he; ++hi) {
        EHCatchScope::Handler handler = catchScope.getHandler(hi);
        assert(handler.Type.Flags == 0 &&
               "landingpads do not support catch handler flags");

        // If this is a catch-all, register that and abort.
        if (!handler.Type.RTTI) {
          assert(!hasCatchAll);
          hasCatchAll = true;
          goto done;
        }

        // Check whether we already have a handler for this type.
        if (catchTypes.insert(handler.Type.RTTI).second) {
          // If not, keep track to later add to catch op.
          clauses.push_back(handler.Type.RTTI);
        }
      }
    }

  done:
    // If we have a catch-all, add null to the landingpad.
    assert(!(hasCatchAll && hasFilter));
    if (hasCatchAll) {
      // Attach the catch_all region. Can't coexist with an unwind one.
      auto catchAll = mlir::cir::CatchAllAttr::get(builder.getContext());
      clauses.push_back(catchAll);

      // If we have an EH filter, we need to add those handlers in the
      // right place in the landingpad, which is to say, at the end.
    } else if (hasFilter) {
      // Create a filter expression: a constant array indicating which filter
      // types there are. The personality routine only lands here if the filter
      // doesn't match.
      llvm_unreachable("NYI");

      // Otherwise, signal that we at least have cleanups.
    } else if (hasCleanup) {
      tryOp.setCleanup(true);
    }

    assert((!clauses.empty() || hasCleanup) && "no catch clauses!");

    // If there's no catch_all, attach the unwind region. This needs to be the
    // last region in the TryOp operation catch list.
    if (!hasCatchAll) {
      auto catchUnwind = mlir::cir::CatchUnwindAttr::get(builder.getContext());
      clauses.push_back(catchUnwind);
    }

    // Add final array of clauses into TryOp.
    tryOp.setCatchTypesAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauses));
  }

  // In traditional LLVM codegen. this tells the backend how to generate the
  // landing pad by generating a branch to the dispatch block. In CIR,
  // getEHDispatchBlock is used to populate blocks for later filing during
  // cleanup handling.
  mlir::Block *dispatch =
      getEHDispatchBlock(EHStack.getInnermostEHScope(), tryOp);
  (void)dispatch;

  return tryOp;
}

// Differently from LLVM traditional codegen, there are no dispatch blocks
// to look at given cir.try_call does not jump to blocks like invoke does.
// However, we keep this around since other parts of CIRGen use
// getCachedEHDispatchBlock to infer state.
mlir::Block *
CIRGenFunction::getEHDispatchBlock(EHScopeStack::stable_iterator si,
                                   mlir::cir::TryOp tryOp) {
  if (EHPersonality::get(*this).usesFuncletPads())
    llvm_unreachable("NYI");

  // The dispatch block for the end of the scope chain is a block that
  // just resumes unwinding.
  if (si == EHStack.stable_end())
    return getEHResumeBlock(true, tryOp);

  // Otherwise, we should look at the actual scope.
  EHScope &scope = *EHStack.find(si);
  auto *dispatchBlock = scope.getCachedEHDispatchBlock();

  mlir::Block *originalBlock = nullptr;
  if (dispatchBlock && tryOp) {
    // If the dispatch is cached but comes from a different tryOp, make sure:
    // - Populate current `tryOp` with a new dispatch block regardless.
    // - Update the map to enqueue new dispatchBlock to also get a cleanup. See
    // code at the end of the function.
    mlir::Operation *parentOp = dispatchBlock->getParentOp();
    if (tryOp != parentOp->getParentOfType<mlir::cir::TryOp>()) {
      originalBlock = dispatchBlock;
      dispatchBlock = nullptr;
    }
  }

  if (!dispatchBlock) {
    switch (scope.getKind()) {
    case EHScope::Catch: {
      // LLVM does some optimization with branches here, CIR just keep track of
      // the corresponding calls.
      assert(callWithExceptionCtx && "expected call information");
      {
        mlir::OpBuilder::InsertionGuard guard(getBuilder());
        assert(callWithExceptionCtx.getCleanup().empty() &&
               "one per call: expected empty region at this point");
        dispatchBlock = builder.createBlock(&callWithExceptionCtx.getCleanup());
        builder.createYield(callWithExceptionCtx.getLoc());
      }
      break;
    }

    case EHScope::Cleanup: {
      if (callWithExceptionCtx && "expected call information") {
        mlir::OpBuilder::InsertionGuard guard(getBuilder());
        assert(callWithExceptionCtx.getCleanup().empty() &&
               "one per call: expected empty region at this point");
        dispatchBlock = builder.createBlock(&callWithExceptionCtx.getCleanup());
        builder.createYield(callWithExceptionCtx.getLoc());
      } else {
        // Usually coming from general cir.scope cleanups that aren't
        // tried to a specific throwing call.
        assert(currLexScope && currLexScope->isRegular() &&
               "expected regular cleanup");
        dispatchBlock = currLexScope->getOrCreateCleanupBlock(builder);
        if (dispatchBlock->empty()) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToEnd(dispatchBlock);
          mlir::Location loc =
              currSrcLoc ? *currSrcLoc : builder.getUnknownLoc();
          builder.createYield(loc);
        }
      }
      break;
    }

    case EHScope::Filter:
      llvm_unreachable("NYI");
      break;

    case EHScope::Terminate:
      llvm_unreachable("NYI");
      break;
    }
  }

  if (originalBlock) {
    // As mentioned above: update the map to enqueue new dispatchBlock to also
    // get a cleanup.
    cleanupsToPatch[originalBlock] = dispatchBlock;
    dispatchBlock = originalBlock;
  } else {
    scope.setCachedEHDispatchBlock(dispatchBlock);
  }
  return dispatchBlock;
}

bool CIRGenFunction::isInvokeDest() {
  if (!EHStack.requiresLandingPad())
    return false;

  // If exceptions are disabled/ignored and SEH is not in use, then there is no
  // invoke destination. SEH "works" even if exceptions are off. In practice,
  // this means that C++ destructors and other EH cleanups don't run, which is
  // consistent with MSVC's behavior, except in the presence of -EHa
  const LangOptions &lo = cgm.getLangOpts();
  if (!lo.Exceptions || lo.IgnoreExceptions) {
    if (!lo.Borland && !lo.MicrosoftExt)
      return false;
    if (!currentFunctionUsesSEHTry())
      return false;
  }

  // CUDA device code doesn't have exceptions.
  if (lo.CUDA && lo.CUDAIsDevice)
    return false;

  return true;
}

mlir::Operation *CIRGenFunction::getInvokeDestImpl(mlir::cir::TryOp tryOp) {
  assert(EHStack.requiresLandingPad());
  assert(!EHStack.empty());
  assert(isInvokeDest());

  // CIR does not cache landing pads.
  const EHPersonality &personality = EHPersonality::get(*this);

  // FIXME(cir): add personality function
  // if (!CurFn->hasPersonalityFn())
  //   CurFn->setPersonalityFn(getOpaquePersonalityFn(CGM, Personality));
  mlir::Operation *lp = nullptr;
  if (personality.usesFuncletPads()) {
    // We don't need separate landing pads in the funclet model.
    llvm::errs() << "PersonalityFn: " << personality.PersonalityFn << "\n";
    llvm_unreachable("NYI");
  } else {
    lp = buildLandingPad(tryOp);
  }

  assert(lp);

  // CIR does not cache landing pads.
  return lp;
}

mlir::Operation *CIRGenFunction::getTerminateLandingPad() {
  llvm_unreachable("NYI");
}
