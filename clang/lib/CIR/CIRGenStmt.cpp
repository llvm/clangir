//===--- CIRGenStmt.cpp - Emit CIR Code from Statements -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

mlir::LogicalResult
CIRGenFunction::buildCompoundStmtWithoutScope(const CompoundStmt &S) {
  for (auto *CurStmt : S.body())
    if (buildStmt(CurStmt, /*useCurrentScope=*/false).failed())
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildCompoundStmt(const CompoundStmt &S) {
  mlir::LogicalResult res = mlir::success();

  auto compoundStmtBuilder = [&]() -> mlir::LogicalResult {
    if (buildCompoundStmtWithoutScope(S).failed())
      return mlir::failure();

    return mlir::success();
  };

  // Add local scope to track new declared variables.
  SymTableScopeTy varScope(symbolTable);
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto locBegin = fusedLoc.getLocations()[0];
        auto locEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{locBegin, locEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexScopeGuard{*this, &lexScope};
        res = compoundStmtBuilder();
      });

  return res;
}

// Build CIR for a statement. useCurrentScope should be true if no
// new scopes need be created when finding a compound statement.
mlir::LogicalResult CIRGenFunction::buildStmt(const Stmt *S,
                                              bool useCurrentScope) {
  if (mlir::succeeded(buildSimpleStmt(S, useCurrentScope)))
    return mlir::success();

  if (getContext().getLangOpts().OpenMP &&
      getContext().getLangOpts().OpenMPSimd)
    assert(0 && "not implemented");

  switch (S->getStmtClass()) {
  default:
    llvm_unreachable("unknown statement class");
  case Stmt::NoStmtClass:
  case Stmt::CXXCatchStmtClass:
  case Stmt::SEHExceptStmtClass:
  case Stmt::SEHFinallyStmtClass:
  case Stmt::MSDependentExistsStmtClass:
    llvm_unreachable("invalid statement class to emit generically");
  case Stmt::NullStmtClass:
  case Stmt::CompoundStmtClass:
  case Stmt::DeclStmtClass:
  case Stmt::LabelStmtClass:
  case Stmt::AttributedStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::DefaultStmtClass:
  case Stmt::CaseStmtClass:
  case Stmt::SEHLeaveStmtClass:
    llvm_unreachable("should have emitted these statements as simple");

#define STMT(Type, Base)
#define ABSTRACT_STMT(Op)
#define EXPR(Type, Base) case Stmt::Type##Class:
#include "clang/AST/StmtNodes.inc"
    {
      // Remember the block we came in on.
      mlir::Block *incoming = builder.getInsertionBlock();
      assert(incoming && "expression emission must have an insertion point");

      buildIgnoredExpr(cast<Expr>(S));

      mlir::Block *outgoing = builder.getInsertionBlock();
      assert(outgoing && "expression emission cleared block!");

      // FIXME: Should we mimic LLVM emission here?
      // The expression emitters assume (reasonably!) that the insertion
      // point is always set.  To maintain that, the call-emission code
      // for noreturn functions has to enter a new block with no
      // predecessors.  We want to kill that block and mark the current
      // insertion point unreachable in the common case of a call like
      // "exit();".  Since expression emission doesn't otherwise create
      // blocks with no predecessors, we can just test for that.
      // However, we must be careful not to do this to our incoming
      // block, because *statement* emission does sometimes create
      // reachable blocks which will have no predecessors until later in
      // the function.  This occurs with, e.g., labels that are not
      // reachable by fallthrough.
      if (incoming != outgoing && outgoing->use_empty())
        assert(0 && "not implemented");
      break;
    }

  case Stmt::IfStmtClass:
    if (buildIfStmt(cast<IfStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::SwitchStmtClass:
    if (buildSwitchStmt(cast<SwitchStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::ForStmtClass:
    if (buildForStmt(cast<ForStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::WhileStmtClass:
    if (buildWhileStmt(cast<WhileStmt>(*S)).failed())
      return mlir::failure();
    break;
  case Stmt::DoStmtClass:
    if (buildDoStmt(cast<DoStmt>(*S)).failed())
      return mlir::failure();
    break;

  case Stmt::IndirectGotoStmtClass:
  case Stmt::ReturnStmtClass:
  // When implemented, GCCAsmStmtClass should fall-through to MSAsmStmtClass.
  case Stmt::GCCAsmStmtClass:
  case Stmt::MSAsmStmtClass:
  case Stmt::CoroutineBodyStmtClass:
  case Stmt::CoreturnStmtClass:
  case Stmt::CapturedStmtClass:
  case Stmt::ObjCAtTryStmtClass:
  case Stmt::ObjCAtThrowStmtClass:
  case Stmt::ObjCAtSynchronizedStmtClass:
  case Stmt::ObjCForCollectionStmtClass:
  case Stmt::ObjCAutoreleasePoolStmtClass:
  case Stmt::CXXTryStmtClass:
  case Stmt::CXXForRangeStmtClass:
  case Stmt::SEHTryStmtClass:
  case Stmt::OMPMetaDirectiveClass:
  case Stmt::OMPCanonicalLoopClass:
  case Stmt::OMPParallelDirectiveClass:
  case Stmt::OMPSimdDirectiveClass:
  case Stmt::OMPTileDirectiveClass:
  case Stmt::OMPUnrollDirectiveClass:
  case Stmt::OMPForDirectiveClass:
  case Stmt::OMPForSimdDirectiveClass:
  case Stmt::OMPSectionsDirectiveClass:
  case Stmt::OMPSectionDirectiveClass:
  case Stmt::OMPSingleDirectiveClass:
  case Stmt::OMPMasterDirectiveClass:
  case Stmt::OMPCriticalDirectiveClass:
  case Stmt::OMPParallelForDirectiveClass:
  case Stmt::OMPParallelForSimdDirectiveClass:
  case Stmt::OMPParallelMasterDirectiveClass:
  case Stmt::OMPParallelSectionsDirectiveClass:
  case Stmt::OMPTaskDirectiveClass:
  case Stmt::OMPTaskyieldDirectiveClass:
  case Stmt::OMPBarrierDirectiveClass:
  case Stmt::OMPTaskwaitDirectiveClass:
  case Stmt::OMPTaskgroupDirectiveClass:
  case Stmt::OMPFlushDirectiveClass:
  case Stmt::OMPDepobjDirectiveClass:
  case Stmt::OMPScanDirectiveClass:
  case Stmt::OMPOrderedDirectiveClass:
  case Stmt::OMPAtomicDirectiveClass:
  case Stmt::OMPTargetDirectiveClass:
  case Stmt::OMPTeamsDirectiveClass:
  case Stmt::OMPCancellationPointDirectiveClass:
  case Stmt::OMPCancelDirectiveClass:
  case Stmt::OMPTargetDataDirectiveClass:
  case Stmt::OMPTargetEnterDataDirectiveClass:
  case Stmt::OMPTargetExitDataDirectiveClass:
  case Stmt::OMPTargetParallelDirectiveClass:
  case Stmt::OMPTargetParallelForDirectiveClass:
  case Stmt::OMPTaskLoopDirectiveClass:
  case Stmt::OMPTaskLoopSimdDirectiveClass:
  case Stmt::OMPMasterTaskLoopDirectiveClass:
  case Stmt::OMPMasterTaskLoopSimdDirectiveClass:
  case Stmt::OMPParallelMasterTaskLoopDirectiveClass:
  case Stmt::OMPParallelMasterTaskLoopSimdDirectiveClass:
  case Stmt::OMPDistributeDirectiveClass:
  case Stmt::OMPTargetUpdateDirectiveClass:
  case Stmt::OMPDistributeParallelForDirectiveClass:
  case Stmt::OMPDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPDistributeSimdDirectiveClass:
  case Stmt::OMPTargetParallelForSimdDirectiveClass:
  case Stmt::OMPTargetSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeDirectiveClass:
  case Stmt::OMPTeamsDistributeSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPTeamsDistributeParallelForDirectiveClass:
  case Stmt::OMPTargetTeamsDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeParallelForDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeParallelForSimdDirectiveClass:
  case Stmt::OMPTargetTeamsDistributeSimdDirectiveClass:
  case Stmt::OMPInteropDirectiveClass:
  case Stmt::OMPDispatchDirectiveClass:
  case Stmt::OMPGenericLoopDirectiveClass:
  case Stmt::OMPMaskedDirectiveClass: {
    llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                 << "' not implemented\n";
    assert(0 && "not implemented");
    break;
  }
  case Stmt::ObjCAtCatchStmtClass:
    llvm_unreachable(
        "@catch statements should be handled by EmitObjCAtTryStmt");
  case Stmt::ObjCAtFinallyStmtClass:
    llvm_unreachable(
        "@finally statements should be handled by EmitObjCAtTryStmt");
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildSimpleStmt(const Stmt *S,
                                                    bool useCurrentScope) {
  switch (S->getStmtClass()) {
  default:
    return mlir::failure();
  case Stmt::DeclStmtClass:
    return buildDeclStmt(cast<DeclStmt>(*S));
  case Stmt::CompoundStmtClass:
    return useCurrentScope
               ? buildCompoundStmtWithoutScope(cast<CompoundStmt>(*S))
               : buildCompoundStmt(cast<CompoundStmt>(*S));
  case Stmt::ReturnStmtClass:
    return buildReturnStmt(cast<ReturnStmt>(*S));
  case Stmt::GotoStmtClass:
    return buildGotoStmt(cast<GotoStmt>(*S));

  case Stmt::NullStmtClass:
    break;

  case Stmt::LabelStmtClass:
    return buildLabelStmt(cast<LabelStmt>(*S));

  case Stmt::CaseStmtClass:
  case Stmt::DefaultStmtClass:
    assert(0 &&
           "Should not get here, currently handled directly from SwitchStmt");
    break;

  case Stmt::BreakStmtClass:
    return buildBreakStmt(cast<BreakStmt>(*S));

  case Stmt::AttributedStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::SEHLeaveStmtClass:
    llvm::errs() << "CIR codegen for '" << S->getStmtClassName()
                 << "' not implemented\n";
    assert(0 && "not implemented");
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildLabelStmt(const clang::LabelStmt &S) {
  if (buildLabel(S.getDecl()).failed())
    return mlir::failure();

  // IsEHa: not implemented.
  assert(!(getContext().getLangOpts().EHAsynch && S.isSideEntry()));

  return buildStmt(S.getSubStmt(), /* useCurrentScope */ true);
}

// Add terminating yield on body regions (loops, ...) in case there are
// not other terminators used.
// FIXME: make terminateCaseRegion use this too.
static void terminateBody(mlir::OpBuilder &builder, mlir::Region &r,
                          mlir::Location loc) {
  if (r.empty())
    return;

  SmallVector<mlir::Block *, 4> eraseBlocks;
  unsigned numBlocks = r.getBlocks().size();
  for (auto &block : r.getBlocks()) {
    // Already cleanup after return operations, which might create
    // empty blocks if emitted as last stmt.
    if (numBlocks != 1 && block.empty() && block.hasNoPredecessors() &&
        block.hasNoSuccessors())
      eraseBlocks.push_back(&block);

    if (block.empty() ||
        !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::OpBuilder::InsertionGuard guardCase(builder);
      builder.setInsertionPointToEnd(&block);
      builder.create<YieldOp>(loc);
    }
  }

  for (auto *b : eraseBlocks)
    b->erase();
}

static mlir::Location getIfLocs(CIRGenFunction &CGF, const clang::Stmt *thenS,
                                const clang::Stmt *elseS) {
  // Attempt to be more accurate as possible with IfOp location, generate
  // one fused location that has either 2 or 4 total locations, depending
  // on else's availability.
  SmallVector<mlir::Location, 4> ifLocs;
  mlir::Attribute metadata;

  clang::SourceRange t = thenS->getSourceRange();
  ifLocs.push_back(CGF.getLoc(t.getBegin()));
  ifLocs.push_back(CGF.getLoc(t.getEnd()));
  if (elseS) {
    clang::SourceRange e = elseS->getSourceRange();
    ifLocs.push_back(CGF.getLoc(e.getBegin()));
    ifLocs.push_back(CGF.getLoc(e.getEnd()));
  }

  return mlir::FusedLoc::get(ifLocs, metadata, CGF.getBuilder().getContext());
}

mlir::LogicalResult CIRGenFunction::buildIfStmt(const IfStmt &S) {
  // The else branch of a consteval if statement is always the only branch
  // that can be runtime evaluated.
  assert(!S.isConsteval() && "not implemented");
  mlir::LogicalResult res = mlir::success();

  // C99 6.8.4.1: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  auto ifStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      buildDecl(*S.getConditionVariable());

    // If the condition constant folds and can be elided, try to avoid
    // emitting the condition and the dead arm of the if/else.
    // FIXME: should this be done as part of a constant folder pass instead?
    bool CondConstant;
    if (ConstantFoldsToSimpleInteger(S.getCond(), CondConstant,
                                     S.isConstexpr())) {
      assert(0 && "not implemented");
    }

    // TODO: PGO and likelihood.
    auto ifLoc = getIfLocs(*this, S.getThen(), S.getElse());
    return buildIfOnBoolExpr(S.getCond(), ifLoc, S.getThen(), S.getElse());
  };

  // TODO: Add a new scoped symbol table.
  // LexicalScope ConditionScope(*this, S.getCond()->getSourceRange());
  // The if scope contains the full source range for IfStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto scopeLocBegin = fusedLoc.getLocations()[0];
        auto scopeLocEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{scopeLocBegin, scopeLocEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexIfScopeGuard{*this, &lexScope};
        res = ifStmtBuilder();
      });

  return res;
}

mlir::LogicalResult CIRGenFunction::buildDeclStmt(const DeclStmt &S) {
  if (!builder.getInsertionBlock()) {
    CGM.emitError("Seems like this is unreachable code, what should we do?");
    return mlir::failure();
  }

  for (const auto *I : S.decls()) {
    buildDecl(*I);
  }

  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildReturnStmt(const ReturnStmt &S) {
  assert(!(getContext().getLangOpts().ElideConstructors &&
           S.getNRVOCandidate() && S.getNRVOCandidate()->isNRVOVariable()) &&
         "unimplemented");
  assert(!FnRetQualTy->isReferenceType() && "unimplemented");
  auto loc = getLoc(S.getSourceRange());

  // Emit the result value, even if unused, to evaluate the side effects.
  const Expr *RV = S.getRetValue();
  if (RV) {
    assert(!isa<ExprWithCleanups>(RV) && "unimplemented");

    mlir::Value V = nullptr;
    switch (CIRGenFunction::getEvaluationKind(RV->getType())) {
    case TEK_Scalar:
      V = buildScalarExpr(RV);
      builder.create<mlir::cir::StoreOp>(loc, V, *FnRetAlloca);
      break;
    case TEK_Complex:
    case TEK_Aggregate:
      llvm::errs() << "ReturnStmt EvaluationKind not implemented\n";
      return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    if (!V || (RV && RV->getType()->isVoidType())) {
      // FIXME: evaluate for side effects.
    }
  } else {
    // Do nothing (return value is left uninitialized), this is also
    // the path when returning from void functions.
  }

  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  builder.create<BrOp>(loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block.
  builder.createBlock(builder.getBlock()->getParent());
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildGotoStmt(const GotoStmt &S) {
  // FIXME: LLVM codegen inserts emit stop point here for debug info
  // sake when the insertion point is available, but doesn't do
  // anything special when there isn't. We haven't implemented debug
  // info support just yet, look at this again once we have it.
  assert(builder.getInsertionBlock() && "not yet implemented");

  // A goto marks the end of a block, create a new one for codegen after
  // buildGotoStmt can resume building in that block.

  // Build a cir.br to the target label.
  auto &JD = LabelMap[S.getLabel()];
  if (buildBranchThroughCleanup(JD, S.getLabel(), getLoc(S.getSourceRange()))
          .failed())
    return mlir::failure();

  // Insert the new block to continue codegen after goto.
  builder.createBlock(builder.getBlock()->getParent());

  // What here...
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildLabel(const LabelDecl *D) {
  JumpDest &Dest = LabelMap[D];

  // Create a new block to tag with a label and add a branch from
  // the current one to it. If the block is empty just call attach it
  // to this label.
  mlir::Block *currBlock = builder.getBlock();
  mlir::Block *labelBlock = currBlock;
  if (!currBlock->empty()) {

    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      labelBlock = builder.createBlock(builder.getBlock()->getParent());
    }

    builder.create<BrOp>(getLoc(D->getSourceRange()), labelBlock);
    builder.setInsertionPointToEnd(labelBlock);
  }

  if (!Dest.isValid()) {
    Dest.Block = labelBlock;
    currLexScope->SolvedLabels.insert(D);
    // FIXME: add a label attribute to block...
  } else {
    assert(0 && "unimplemented");
  }

  //  FIXME: emit debug info for labels, incrementProfileCounter
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildBreakStmt(const clang::BreakStmt &S) {
  builder.create<YieldOp>(
      getLoc(S.getBreakLoc()),
      mlir::cir::YieldOpKindAttr::get(builder.getContext(),
                                      mlir::cir::YieldOpKind::Break),
      mlir::ValueRange({}));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildCaseStmt(const CaseStmt &S,
                                                  mlir::Type condType,
                                                  CaseAttr &caseEntry) {
  assert((!S.getRHS() || !S.caseStmtIsGNURange()) &&
         "case ranges not implemented");
  auto res = mlir::success();

  const CaseStmt *caseStmt = &S;
  SmallVector<mlir::Attribute, 4> caseEltValueListAttr;
  // Fold cascading cases whenever possible to simplify codegen a bit.
  while (true) {
    auto intVal = caseStmt->getLHS()->EvaluateKnownConstInt(getContext());
    caseEltValueListAttr.push_back(mlir::IntegerAttr::get(condType, intVal));
    if (isa<CaseStmt>(caseStmt->getSubStmt()))
      caseStmt = dyn_cast_or_null<CaseStmt>(caseStmt->getSubStmt());
    else
      break;
  }

  auto caseValueList = builder.getArrayAttr(caseEltValueListAttr);

  auto *ctx = builder.getContext();
  caseEntry = mlir::cir::CaseAttr::get(
      ctx, caseValueList,
      CaseOpKindAttr::get(ctx, caseEltValueListAttr.size() > 1
                                   ? mlir::cir::CaseOpKind::Anyof
                                   : mlir::cir::CaseOpKind::Equal));

  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    res = buildStmt(
        caseStmt->getSubStmt(),
        /*useCurrentScope=*/!isa<CompoundStmt>(caseStmt->getSubStmt()));
  }

  // TODO: likelihood
  return res;
}

mlir::LogicalResult CIRGenFunction::buildDefaultStmt(const DefaultStmt &S,
                                                     mlir::Type condType,
                                                     CaseAttr &caseEntry) {
  auto res = mlir::success();
  auto *ctx = builder.getContext();
  caseEntry = mlir::cir::CaseAttr::get(
      ctx, builder.getArrayAttr({}),
      CaseOpKindAttr::get(ctx, mlir::cir::CaseOpKind::Default));
  {
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    res = buildStmt(S.getSubStmt(),
                    /*useCurrentScope=*/!isa<CompoundStmt>(S.getSubStmt()));
  }

  // TODO: likelihood
  return res;
}

mlir::LogicalResult CIRGenFunction::buildForStmt(const ForStmt &S) {
  mlir::cir::LoopOp loopOp;

  // TODO: pass in array of attributes.
  auto forStmtBuilder = [&]() -> mlir::LogicalResult {
    auto forRes = mlir::success();
    // Evaluate the first part before the loop.
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    loopOp = builder.create<LoopOp>(
        getLoc(S.getSourceRange()), mlir::cir::LoopOpKind::For,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // TODO: branch weigths, likelyhood, profile counter, etc.
          mlir::Value condVal;
          if (S.getCond()) {
            // If the for statement has a condition scope,
            // emit the local variable declaration.
            if (S.getConditionVariable())
              buildDecl(*S.getConditionVariable());
            // C99 6.8.5p2/p4: The first substatement is executed if the
            // expression compares unequal to 0. The condition must be a
            // scalar type.
            condVal = evaluateExprAsBool(S.getCond());
          } else {
            condVal = b.create<mlir::cir::ConstantOp>(
                loc, mlir::cir::BoolType::get(b.getContext()),
                b.getBoolAttr(true));
          }
          b.create<YieldOp>(loc, condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // FIXME: in C we need to open a new scope here. Do we also need it
          // for C++ in case it's a compound statement?
          if (buildStmt(S.getBody(), /*useCurrentScope=*/true).failed())
            forRes = mlir::failure();
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (S.getInc())
            if (buildStmt(S.getInc(), /*useCurrentScope=*/true).failed())
              forRes = mlir::failure();
          builder.create<YieldOp>(loc);
        });
    return forRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto scopeLocBegin = fusedLoc.getLocations()[0];
        auto scopeLocEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{scopeLocBegin, scopeLocEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexForScopeGuard{*this, &lexScope};
        res = forStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, loopOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildDoStmt(const DoStmt &S) {
  mlir::cir::LoopOp loopOp;

  // TODO: pass in array of attributes.
  auto doStmtBuilder = [&]() -> mlir::LogicalResult {
    auto forRes = mlir::success();

    loopOp = builder.create<LoopOp>(
        getLoc(S.getSourceRange()), mlir::cir::LoopOpKind::DoWhile,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // TODO: branch weigths, likelyhood, profile counter, etc.
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          mlir::Value condVal = evaluateExprAsBool(S.getCond());
          b.create<YieldOp>(loc, condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (buildStmt(S.getBody(), /*useCurrentScope=*/true).failed())
            forRes = mlir::failure();
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          builder.create<YieldOp>(loc);
        });
    return forRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto scopeLocBegin = fusedLoc.getLocations()[0];
        auto scopeLocEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{scopeLocBegin, scopeLocEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexForScopeGuard{*this, &lexScope};
        res = doStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, loopOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildWhileStmt(const WhileStmt &S) {
  mlir::cir::LoopOp loopOp;

  // TODO: pass in array of attributes.
  auto whileStmtBuilder = [&]() -> mlir::LogicalResult {
    auto forRes = mlir::success();

    loopOp = builder.create<LoopOp>(
        getLoc(S.getSourceRange()), mlir::cir::LoopOpKind::While,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          // TODO: branch weigths, likelyhood, profile counter, etc.
          mlir::Value condVal;
          // If the for statement has a condition scope,
          // emit the local variable declaration.
          if (S.getConditionVariable())
            buildDecl(*S.getConditionVariable());
          // C99 6.8.5p2/p4: The first substatement is executed if the
          // expression compares unequal to 0. The condition must be a
          // scalar type.
          condVal = evaluateExprAsBool(S.getCond());
          b.create<YieldOp>(loc, condVal);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          if (buildStmt(S.getBody(), /*useCurrentScope=*/true).failed())
            forRes = mlir::failure();
        },
        /*stepBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          builder.create<YieldOp>(loc);
        });
    return forRes;
  };

  auto res = mlir::success();
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto scopeLocBegin = fusedLoc.getLocations()[0];
        auto scopeLocEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{scopeLocBegin, scopeLocEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexForScopeGuard{*this, &lexScope};
        res = whileStmtBuilder();
      });

  if (res.failed())
    return res;

  terminateBody(builder, loopOp.getBody(), getLoc(S.getEndLoc()));
  return mlir::success();
}

mlir::LogicalResult CIRGenFunction::buildSwitchStmt(const SwitchStmt &S) {
  // TODO: LLVM codegen does some early optimization to fold the condition and
  // only emit live cases. CIR should use MLIR to achieve similar things,
  // nothing to be done here.
  // if (ConstantFoldsToSimpleInteger(S.getCond(), ConstantCondValue))...

  auto res = mlir::success();
  SwitchOp swop;

  auto switchStmtBuilder = [&]() -> mlir::LogicalResult {
    if (S.getInit())
      if (buildStmt(S.getInit(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

    if (S.getConditionVariable())
      buildDecl(*S.getConditionVariable());

    mlir::Value condV = buildScalarExpr(S.getCond());

    // TODO: PGO and likelihood (e.g. PGO.haveRegionCounts())
    // TODO: if the switch has a condition wrapped by __builtin_unpredictable?

    // FIXME: track switch to handle nested stmts.
    swop = builder.create<SwitchOp>(
        getLoc(S.getBeginLoc()), condV,
        /*switchBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc, mlir::OperationState &os) {
          auto *cs = dyn_cast<CompoundStmt>(S.getBody());
          assert(cs && "expected compound stmt");
          SmallVector<mlir::Attribute, 4> caseAttrs;

          currLexScope->setAsSwitch();
          mlir::Block *lastCaseBlock = nullptr;
          for (auto *c : cs->body()) {
            bool caseLike = isa<CaseStmt, DefaultStmt>(c);
            if (!caseLike) {
              // This means it's a random stmt following up a case, just
              // emit it as part of previous known case.
              assert(lastCaseBlock && "expects pre-existing case block");
              mlir::OpBuilder::InsertionGuard guardCase(builder);
              builder.setInsertionPointToEnd(lastCaseBlock);
              res = buildStmt(c, /*useCurrentScope=*/!isa<CompoundStmt>(c));
              if (res.failed())
                break;
              continue;
            }

            auto *caseStmt = dyn_cast<CaseStmt>(c);
            CaseAttr caseAttr;
            {
              mlir::OpBuilder::InsertionGuard guardCase(builder);

              // Update scope information with the current region we are
              // emitting code for. This is useful to allow return blocks to be
              // automatically and properly placed during cleanup.
              mlir::Region *caseRegion = os.addRegion();
              currLexScope->updateCurrentSwitchCaseRegion();

              lastCaseBlock = builder.createBlock(caseRegion);
              if (caseStmt)
                res = buildCaseStmt(*caseStmt, condV.getType(), caseAttr);
              else {
                auto *defaultStmt = dyn_cast<DefaultStmt>(c);
                assert(defaultStmt && "expected default stmt");
                res = buildDefaultStmt(*defaultStmt, condV.getType(), caseAttr);
              }

              if (res.failed())
                break;
            }
            caseAttrs.push_back(caseAttr);
          }

          os.addAttribute("cases", builder.getArrayAttr(caseAttrs));
        });

    if (res.failed())
      return res;
    return mlir::success();
  };

  // The switch scope contains the full source range for SwitchStmt.
  auto scopeLoc = getLoc(S.getSourceRange());
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, mlir::TypeRange(), /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto fusedLoc = loc.cast<mlir::FusedLoc>();
        auto scopeLocBegin = fusedLoc.getLocations()[0];
        auto scopeLocEnd = fusedLoc.getLocations()[1];
        LexicalScopeContext lexScope{scopeLocBegin, scopeLocEnd,
                                     builder.getInsertionBlock()};
        LexicalScopeGuard lexIfScopeGuard{*this, &lexScope};
        res = switchStmtBuilder();
      });

  if (res.failed())
    return res;

  // Any block in a case region without a terminator is considered a
  // fallthrough yield. In practice there shouldn't be more than one
  // block without a terminator, we patch any block we see though and
  // let mlir's SwitchOp verifier enforce rules.
  auto terminateCaseRegion = [&](mlir::Region &r, mlir::Location loc) {
    if (r.empty())
      return;

    SmallVector<mlir::Block *, 4> eraseBlocks;
    unsigned numBlocks = r.getBlocks().size();
    for (auto &block : r.getBlocks()) {
      // Already cleanup after return operations, which might create
      // empty blocks if emitted as last stmt.
      if (numBlocks != 1 && block.empty() && block.hasNoPredecessors() &&
          block.hasNoSuccessors())
        eraseBlocks.push_back(&block);

      if (block.empty() ||
          !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::OpBuilder::InsertionGuard guardCase(builder);
        builder.setInsertionPointToEnd(&block);
        builder.create<YieldOp>(
            loc,
            mlir::cir::YieldOpKindAttr::get(
                builder.getContext(), mlir::cir::YieldOpKind::Fallthrough),
            mlir::ValueRange({}));
      }
    }

    for (auto *b : eraseBlocks)
      b->erase();
  };

  // Make sure all case regions are terminated by inserting fallthroughs
  // when necessary.
  // FIXME: find a better way to get accurante with location here.
  for (auto &r : swop.getRegions())
    terminateCaseRegion(r, swop.getLoc());
  return mlir::success();
}
