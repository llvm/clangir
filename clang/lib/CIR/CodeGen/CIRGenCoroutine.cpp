//===----- CGCoroutine.cpp - Emit CIR Code for C++ coroutines -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of coroutines.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "CIRGenFunction.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace cir;

struct cir::CGCoroData {
  // What is the current await expression kind and how many
  // await/yield expressions were encountered so far.
  // These are used to generate pretty labels for await expressions in LLVM IR.
  mlir::cir::AwaitKind currentAwaitKind = mlir::cir::AwaitKind::init;

  // Stores the __builtin_coro_id emitted in the function so that we can supply
  // it as the first argument to other builtins.
  mlir::cir::CallOp coroId = nullptr;

  // Stores the result of __builtin_coro_begin call.
  mlir::Value coroBegin = nullptr;

  // Stores the insertion point for final suspend, this happens after the
  // promise call (return_xxx promise member) but before a cir.br to the return
  // block.
  mlir::Operation *finalSuspendInsPoint;

  // How many co_return statements are in the coroutine. Used to decide whether
  // we need to add co_return; equivalent at the end of the user authored body.
  unsigned coreturnCount = 0;

  // The promise type's 'unhandled_exception' handler, if it defines one.
  Stmt *exceptionHandler = nullptr;
};

// Defining these here allows to keep CGCoroData private to this file.
CIRGenFunction::CGCoroInfo::CGCoroInfo() = default;
CIRGenFunction::CGCoroInfo::~CGCoroInfo() = default;

static void createCoroData(CIRGenFunction &cgf,
                           CIRGenFunction::CGCoroInfo &curCoro,
                           mlir::cir::CallOp coroId) {
  if (curCoro.Data) {
    llvm_unreachable("EmitCoroutineBodyStatement called twice?");

    return;
  }

  curCoro.Data = std::make_unique<CGCoroData>();
  curCoro.Data->coroId = coroId;
}

namespace {
// FIXME: both GetParamRef and ParamReferenceReplacerRAII are good template
// candidates to be shared among LLVM / CIR codegen.

// Hunts for the parameter reference in the parameter copy/move declaration.
struct GetParamRef : public StmtVisitor<GetParamRef> {
public:
  DeclRefExpr *expr = nullptr;
  GetParamRef() = default;
  void VisitDeclRefExpr(DeclRefExpr *e) {
    assert(expr == nullptr && "multilple declref in param move");
    expr = e;
  }
  void VisitStmt(Stmt *s) {
    for (auto *c : s->children()) {
      if (c)
        Visit(c);
    }
  }
};

// This class replaces references to parameters to their copies by changing
// the addresses in CGF.LocalDeclMap and restoring back the original values in
// its destructor.
struct ParamReferenceReplacerRAII {
  CIRGenFunction::DeclMapTy savedLocals;
  CIRGenFunction::DeclMapTy &localDeclMap;

  ParamReferenceReplacerRAII(CIRGenFunction::DeclMapTy &localDeclMap)
      : localDeclMap(localDeclMap) {}

  void addCopy(DeclStmt const *pm) {
    // Figure out what param it refers to.

    assert(pm->isSingleDecl());
    VarDecl const *vd = static_cast<VarDecl const *>(pm->getSingleDecl());
    Expr const *initExpr = vd->getInit();
    GetParamRef visitor;
    visitor.Visit(const_cast<Expr *>(initExpr));
    assert(visitor.expr);
    DeclRefExpr *dreOrig = visitor.expr;
    auto *pd = dreOrig->getDecl();

    auto it = localDeclMap.find(pd);
    assert(it != localDeclMap.end() && "parameter is not found");
    savedLocals.insert({pd, it->second});

    auto copyIt = localDeclMap.find(vd);
    assert(copyIt != localDeclMap.end() && "parameter copy is not found");
    it->second = copyIt->getSecond();
  }

  ~ParamReferenceReplacerRAII() {
    for (auto &&savedLocal : savedLocals) {
      localDeclMap.insert({savedLocal.first, savedLocal.second});
    }
  }
};
} // namespace

// Emit coroutine intrinsic and patch up arguments of the token type.
RValue CIRGenFunction::buildCoroutineIntrinsic(const CallExpr *e,
                                               unsigned int iid) {
  llvm_unreachable("NYI");
}

RValue CIRGenFunction::buildCoroutineFrame() {
  if (CurCoro.Data && CurCoro.Data->coroBegin) {
    return RValue::get(CurCoro.Data->coroBegin);
  }
  llvm_unreachable("NYI");
}

static mlir::LogicalResult
buildBodyAndFallthrough(CIRGenFunction &cgf, const CoroutineBodyStmt &s,
                        Stmt *body,
                        const CIRGenFunction::LexicalScope *currLexScope) {
  if (cgf.buildStmt(body, /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Note that LLVM checks CanFallthrough by looking into the availability
  // of the insert block which is kinda brittle and unintuitive, seems to be
  // related with how landing pads are handled.
  //
  // CIRGen handles this by checking pre-existing co_returns in the current
  // scope instead. Are we missing anything?
  //
  // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
  const bool canFallthrough = !currLexScope->hasCoreturn();
  if (canFallthrough)
    if (Stmt *onFallthrough = s.getFallthroughHandler())
      if (cgf.buildStmt(onFallthrough, /*useCurrentScope=*/true).failed())
        return mlir::failure();

  return mlir::success();
}

mlir::cir::CallOp CIRGenFunction::buildCoroIDBuiltinCall(mlir::Location loc,
                                                         mlir::Value nullPtr) {
  auto int32Ty = builder.getUInt32Ty();

  auto &ti = CGM.getASTContext().getTargetInfo();
  unsigned newAlign = ti.getNewAlign() / ti.getCharWidth();

  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroId);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroId,
        mlir::cir::FuncType::get({int32Ty, VoidPtrTy, VoidPtrTy, VoidPtrTy},
                                 int32Ty),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.createCallOp(loc, fnOp,
                              mlir::ValueRange{builder.getUInt32(newAlign, loc),
                                               nullPtr, nullPtr, nullPtr});
}

mlir::cir::CallOp
CIRGenFunction::buildCoroAllocBuiltinCall(mlir::Location loc) {
  auto boolTy = builder.getBoolTy();
  auto int32Ty = builder.getUInt32Ty();

  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroAlloc);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(loc, CGM.builtinCoroAlloc,
                                 mlir::cir::FuncType::get({int32Ty}, boolTy),
                                 /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.createCallOp(
      loc, fnOp, mlir::ValueRange{CurCoro.Data->coroId.getResult()});
}

mlir::cir::CallOp
CIRGenFunction::buildCoroBeginBuiltinCall(mlir::Location loc,
                                          mlir::Value coroframeAddr) {
  auto int32Ty = builder.getUInt32Ty();
  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroBegin);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroBegin,
        mlir::cir::FuncType::get({int32Ty, VoidPtrTy}, VoidPtrTy),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.createCallOp(
      loc, fnOp,
      mlir::ValueRange{CurCoro.Data->coroId.getResult(), coroframeAddr});
}

mlir::cir::CallOp CIRGenFunction::buildCoroEndBuiltinCall(mlir::Location loc,
                                                          mlir::Value nullPtr) {
  auto boolTy = builder.getBoolTy();
  mlir::Operation *builtin = CGM.getGlobalValue(CGM.builtinCoroEnd);

  mlir::cir::FuncOp fnOp;
  if (!builtin) {
    fnOp = CGM.createCIRFunction(
        loc, CGM.builtinCoroEnd,
        mlir::cir::FuncType::get({VoidPtrTy, boolTy}, boolTy),
        /*FD=*/nullptr);
    assert(fnOp && "should always succeed");
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
  } else
    fnOp = cast<mlir::cir::FuncOp>(builtin);

  return builder.createCallOp(
      loc, fnOp, mlir::ValueRange{nullPtr, builder.getBool(false, loc)});
}

mlir::LogicalResult
CIRGenFunction::buildCoroutineBody(const CoroutineBodyStmt &s) {
  auto openCurlyLoc = getLoc(s.getBeginLoc());
  auto nullPtrCst = builder.getNullPtr(VoidPtrTy, openCurlyLoc);

  auto fn = dyn_cast<mlir::cir::FuncOp>(CurFn);
  assert(fn && "other callables NYI");
  fn.setCoroutineAttr(mlir::UnitAttr::get(builder.getContext()));
  auto coroId = buildCoroIDBuiltinCall(openCurlyLoc, nullPtrCst);
  createCoroData(*this, CurCoro, coroId);

  // Backend is allowed to elide memory allocations, to help it, emit
  // auto mem = coro.alloc() ? 0 : ... allocation code ...;
  auto coroAlloc = buildCoroAllocBuiltinCall(openCurlyLoc);

  // Initialize address of coroutine frame to null
  auto astVoidPtrTy = CGM.getASTContext().VoidPtrTy;
  auto allocaTy = getTypes().convertTypeForMem(astVoidPtrTy);
  Address coroFrame =
      CreateTempAlloca(allocaTy, getContext().getTypeAlignInChars(astVoidPtrTy),
                       openCurlyLoc, "__coro_frame_addr",
                       /*ArraySize=*/nullptr);

  auto storeAddr = coroFrame.getPointer();
  builder.CIRBaseBuilderTy::createStore(openCurlyLoc, nullPtrCst, storeAddr);
  builder.create<mlir::cir::IfOp>(openCurlyLoc, coroAlloc.getResult(),
                                  /*withElseRegion=*/false,
                                  /*thenBuilder=*/
                                  [&](mlir::OpBuilder &b, mlir::Location loc) {
                                    builder.CIRBaseBuilderTy::createStore(
                                        loc, buildScalarExpr(s.getAllocate()),
                                        storeAddr);
                                    builder.create<mlir::cir::YieldOp>(loc);
                                  });

  CurCoro.Data->coroBegin =
      buildCoroBeginBuiltinCall(
          openCurlyLoc,
          builder.create<mlir::cir::LoadOp>(openCurlyLoc, allocaTy, storeAddr))
          .getResult();

  // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
  if (auto *retOnAllocFailure = s.getReturnStmtOnAllocFailure())
    llvm_unreachable("NYI");

  {
    // FIXME(cir): create a new scope to copy out the params?
    // LLVM create scope cleanups here, but might be due to the use
    // of many basic blocks?
    assert(!MissingFeatures::generateDebugInfo() && "NYI");
    ParamReferenceReplacerRAII paramReplacer(LocalDeclMap);

    // Create mapping between parameters and copy-params for coroutine
    // function.
    llvm::ArrayRef<const Stmt *> ParamMoves = s.getParamMoves();
    assert((ParamMoves.empty() || (ParamMoves.size() == FnArgs.size())) &&
           "ParamMoves and FnArgs should be the same size for coroutine "
           "function");
    // For zipping the arg map into debug info.
    assert(!MissingFeatures::generateDebugInfo() && "NYI");

    // Create parameter copies. We do it before creating a promise, since an
    // evolution of coroutine TS may allow promise constructor to observe
    // parameter copies.
    for (auto *pm : s.getParamMoves()) {
      if (buildStmt(pm, /*useCurrentScope=*/true).failed())
        return mlir::failure();
      paramReplacer.addCopy(cast<DeclStmt>(pm));
    }

    if (buildStmt(s.getPromiseDeclStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    // ReturnValue should be valid as long as the coroutine's return type
    // is not void. The assertion could help us to reduce the check later.
    assert(ReturnValue.isValid() == (bool)s.getReturnStmt());
    // Now we have the promise, initialize the GRO.
    // We need to emit `get_return_object` first. According to:
    // [dcl.fct.def.coroutine]p7
    // The call to get_return_­object is sequenced before the call to
    // initial_suspend and is invoked at most once.
    //
    // So we couldn't emit return value when we emit return statment,
    // otherwise the call to get_return_object wouldn't be in front
    // of initial_suspend.
    if (ReturnValue.isValid()) {
      buildAnyExprToMem(s.getReturnValue(), ReturnValue,
                        s.getReturnValue()->getType().getQualifiers(),
                        /*IsInit*/ true);
    }

    // FIXME(cir): EHStack.pushCleanup<CallCoroEnd>(EHCleanup);
    CurCoro.Data->currentAwaitKind = mlir::cir::AwaitKind::init;
    if (buildStmt(s.getInitSuspendStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    CurCoro.Data->currentAwaitKind = mlir::cir::AwaitKind::user;

    // FIXME(cir): wrap buildBodyAndFallthrough with try/catch bits.
    if (s.getExceptionHandler())
      assert(!MissingFeatures::unhandledException() && "NYI");
    if (buildBodyAndFallthrough(*this, s, s.getBody(), currLexScope).failed())
      return mlir::failure();

    // Note that LLVM checks CanFallthrough by looking into the availability
    // of the insert block which is kinda brittle and unintuitive, seems to be
    // related with how landing pads are handled.
    //
    // CIRGen handles this by checking pre-existing co_returns in the current
    // scope instead. Are we missing anything?
    //
    // From LLVM IR Gen: const bool CanFallthrough = Builder.GetInsertBlock();
    const bool canFallthrough = currLexScope->hasCoreturn();
    const bool hasCoreturns = CurCoro.Data->coreturnCount > 0;
    if (canFallthrough || hasCoreturns) {
      CurCoro.Data->currentAwaitKind = mlir::cir::AwaitKind::final;
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(CurCoro.Data->finalSuspendInsPoint);
        if (buildStmt(s.getFinalSuspendStmt(), /*useCurrentScope=*/true)
                .failed())
          return mlir::failure();
      }
    }
  }
  return mlir::success();
}

static bool memberCallExpressionCanThrow(const Expr *e) {
  if (const auto *ce = dyn_cast<CXXMemberCallExpr>(e))
    if (const auto *proto =
            ce->getMethodDecl()->getType()->getAs<FunctionProtoType>())
      if (isNoexceptExceptionSpec(proto->getExceptionSpecType()) &&
          proto->canThrow() == CT_Cannot)
        return false;
  return true;
}

// Given a suspend expression which roughly looks like:
//
//   auto && x = CommonExpr();
//   if (!x.await_ready()) {
//      x.await_suspend(...); (*)
//   }
//   x.await_resume();
//
// where the result of the entire expression is the result of x.await_resume()
//
//   (*) If x.await_suspend return type is bool, it allows to veto a suspend:
//      if (x.await_suspend(...))
//        llvm_coro_suspend();
//
// This is more higher level than LLVM codegen, for that one see llvm's
// docs/Coroutines.rst for more details.
namespace {
struct LValueOrRValue {
  LValue lv;
  RValue rv;
};
} // namespace
static LValueOrRValue
buildSuspendExpression(CIRGenFunction &cgf, CGCoroData &coro,
                       CoroutineSuspendExpr const &s, mlir::cir::AwaitKind kind,
                       AggValueSlot aggSlot, bool ignoreResult,
                       mlir::Block *scopeParentBlock,
                       mlir::Value &tmpResumeRValAddr, bool forLValue) {
  auto *e = s.getCommonExpr();

  auto awaitBuild = mlir::success();
  LValueOrRValue awaitRes;

  auto binder =
      CIRGenFunction::OpaqueValueMappingData::bind(cgf, s.getOpaqueValue(), e);
  auto unbindOnExit = llvm::make_scope_exit([&] { binder.unbind(cgf); });
  auto &builder = cgf.getBuilder();

  [[maybe_unused]] auto awaitOp = builder.create<mlir::cir::AwaitOp>(
      cgf.getLoc(s.getSourceRange()), kind,
      /*readyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        Expr *condExpr = s.getReadyExpr()->IgnoreParens();
        builder.createCondition(cgf.evaluateExprAsBool(condExpr));
      },
      /*suspendBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Note that differently from LLVM codegen we do not emit coro.save
        // and coro.suspend here, that should be done as part of lowering this
        // to LLVM dialect (or some other MLIR dialect)

        // A invalid suspendRet indicates "void returning await_suspend"
        auto suspendRet = cgf.buildScalarExpr(s.getSuspendExpr());

        // Veto suspension if requested by bool returning await_suspend.
        if (suspendRet) {
          // From LLVM codegen:
          // if (SuspendRet != nullptr && SuspendRet->getType()->isIntegerTy(1))
          llvm_unreachable("NYI");
        }

        // Signals the parent that execution flows to next region.
        builder.create<mlir::cir::YieldOp>(loc);
      },
      /*resumeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        // Exception handling requires additional IR. If the 'await_resume'
        // function is marked as 'noexcept', we avoid generating this additional
        // IR.
        CXXTryStmt *tryStmt = nullptr;
        if (coro.exceptionHandler && kind == mlir::cir::AwaitKind::init &&
            memberCallExpressionCanThrow(s.getResumeExpr())) {
          llvm_unreachable("NYI");
        }

        // FIXME(cir): the alloca for the resume expr should be placed in the
        // enclosing cir.scope instead.
        if (forLValue)
          awaitRes.lv = cgf.buildLValue(s.getResumeExpr());
        else {
          awaitRes.rv =
              cgf.buildAnyExpr(s.getResumeExpr(), aggSlot, ignoreResult);
          if (!awaitRes.rv.isIgnored()) {
            // Create the alloca in the block before the scope wrapping
            // cir.await.
            tmpResumeRValAddr = cgf.buildAlloca(
                "__coawait_resume_rval", awaitRes.rv.getScalarVal().getType(),
                loc, CharUnits::One(),
                builder.getBestAllocaInsertPoint(scopeParentBlock));
            // Store the rvalue so we can reload it before the promise call.
            builder.CIRBaseBuilderTy::createStore(
                loc, awaitRes.rv.getScalarVal(), tmpResumeRValAddr);
          }
        }

        if (tryStmt) {
          llvm_unreachable("NYI");
        }

        // Returns control back to parent.
        builder.create<mlir::cir::YieldOp>(loc);
      });

  assert(awaitBuild.succeeded() && "Should know how to codegen");
  return awaitRes;
}

static RValue buildSuspendExpr(CIRGenFunction &cgf,
                               const CoroutineSuspendExpr &e,
                               mlir::cir::AwaitKind kind, AggValueSlot aggSlot,
                               bool ignoreResult) {
  RValue rval;
  auto scopeLoc = cgf.getLoc(e.getSourceRange());

  // Since we model suspend / resume as an inner region, we must store
  // resume scalar results in a tmp alloca, and load it after we build the
  // suspend expression. An alternative way to do this would be to make
  // every region return a value when promise.return_value() is used, but
  // it's a bit awkward given that resume is the only region that actually
  // returns a value.
  mlir::Block *currEntryBlock = cgf.currLexScope->getEntryBlock();
  [[maybe_unused]] mlir::Value tmpResumeRValAddr;

  // No need to explicitly wrap this into a scope since the AST already uses a
  // ExprWithCleanups, which will wrap this into a cir.scope anyways.
  rval = buildSuspendExpression(cgf, *cgf.CurCoro.Data, e, kind, aggSlot,
                                ignoreResult, currEntryBlock, tmpResumeRValAddr,
                                /*forLValue*/ false)
             .rv;

  if (ignoreResult || rval.isIgnored())
    return rval;

  if (rval.isScalar()) {
    rval = RValue::get(cgf.getBuilder().create<mlir::cir::LoadOp>(
        scopeLoc, rval.getScalarVal().getType(), tmpResumeRValAddr));
  } else if (rval.isAggregate()) {
    // This is probably already handled via AggSlot, remove this assertion
    // once we have a testcase and prove all pieces work.
    llvm_unreachable("NYI");
  } else { // complex
    llvm_unreachable("NYI");
  }
  return rval;
}

RValue CIRGenFunction::buildCoawaitExpr(const CoawaitExpr &e,
                                        AggValueSlot aggSlot,
                                        bool ignoreResult) {
  return buildSuspendExpr(*this, e, CurCoro.Data->currentAwaitKind, aggSlot,
                          ignoreResult);
}

RValue CIRGenFunction::buildCoyieldExpr(const CoyieldExpr &e,
                                        AggValueSlot aggSlot,
                                        bool ignoreResult) {
  return buildSuspendExpr(*this, e, mlir::cir::AwaitKind::yield, aggSlot,
                          ignoreResult);
}

mlir::LogicalResult CIRGenFunction::buildCoreturnStmt(CoreturnStmt const &s) {
  ++CurCoro.Data->coreturnCount;
  currLexScope->setCoreturn();

  const Expr *rv = s.getOperand();
  if (rv && rv->getType()->isVoidType() && !isa<InitListExpr>(rv)) {
    // Make sure to evaluate the non initlist expression of a co_return
    // with a void expression for side effects.
    // FIXME(cir): add scope
    // RunCleanupsScope cleanupScope(*this);
    buildIgnoredExpr(rv);
  }
  if (buildStmt(s.getPromiseCall(), /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // Create a new return block (if not existent) and add a branch to
  // it. The actual return instruction is only inserted during current
  // scope cleanup handling.
  auto loc = getLoc(s.getSourceRange());
  auto *retBlock = currLexScope->getOrCreateRetBlock(*this, loc);
  CurCoro.Data->finalSuspendInsPoint =
      builder.create<mlir::cir::BrOp>(loc, retBlock);

  // Insert the new block to continue codegen after branch to ret block,
  // this will likely be an empty block.
  builder.createBlock(builder.getBlock()->getParent());

  // TODO(cir): LLVM codegen for a cleanup on cleanupScope here.
  return mlir::success();
}
