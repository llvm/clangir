//===--- CIRGenCall.cpp - Encapsulate calling convention details ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenFunctionInfo.h"
#include "CIRGenTypes.h"
#include "TargetInfo.h"

#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/FnInfoOpts.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "clang/CIR/MissingFeatures.h"

using namespace cir;
using namespace clang;

CIRGenFunctionInfo *CIRGenFunctionInfo::create(
    mlir::cir::CallingConv cirCC, bool instanceMethod, bool chainCall,
    const FunctionType::ExtInfo &info,
    llvm::ArrayRef<ExtParameterInfo> paramInfos, CanQualType resultType,
    llvm::ArrayRef<CanQualType> argTypes, RequiredArgs required) {
  assert(paramInfos.empty() || paramInfos.size() == argTypes.size());
  assert(!required.allowsOptionalArgs() ||
         required.getNumRequiredArgs() <= argTypes.size());

  void *buffer = operator new(totalSizeToAlloc<ArgInfo, ExtParameterInfo>(
      argTypes.size() + 1, paramInfos.size()));

  CIRGenFunctionInfo *fi = new (buffer) CIRGenFunctionInfo();
  fi->CallingConvention = cirCC;
  fi->EffectiveCallingConvention = cirCC;
  fi->ASTCallingConvention = info.getCC();
  fi->InstanceMethod = instanceMethod;
  fi->ChainCall = chainCall;
  fi->CmseNSCall = info.getCmseNSCall();
  fi->NoReturn = info.getNoReturn();
  fi->ReturnsRetained = info.getProducesResult();
  fi->NoCallerSavedRegs = info.getNoCallerSavedRegs();
  fi->NoCfCheck = info.getNoCfCheck();
  fi->Required = required;
  fi->HasRegParm = info.getHasRegParm();
  fi->RegParm = info.getRegParm();
  fi->ArgStruct = nullptr;
  fi->ArgStructAlign = 0;
  fi->NumArgs = argTypes.size();
  fi->HasExtParameterInfos = !paramInfos.empty();
  fi->getArgsBuffer()[0].type = resultType;
  for (unsigned i = 0; i < argTypes.size(); ++i)
    fi->getArgsBuffer()[i + 1].type = argTypes[i];
  for (unsigned i = 0; i < paramInfos.size(); ++i)
    fi->getExtParameterInfosBuffer()[i] = paramInfos[i];

  return fi;
}

namespace {

/// Encapsulates information about the way function arguments from
/// CIRGenFunctionInfo should be passed to actual CIR function.
class ClangToCIRArgMapping {
  static const unsigned invalidIndex = ~0U;
  unsigned inallocaArgNo;
  unsigned sRetArgNo;
  unsigned totalCIRArguments = 0;

  /// Arguments of CIR function corresponding to single Clang argument.
  struct CIRArgs {
    unsigned paddingArgIndex = 0;
    // Argument is expanded to CIR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned firstArgIndex = 0;
    unsigned numberOfArgs = 0;

    CIRArgs() : paddingArgIndex(invalidIndex), firstArgIndex(invalidIndex) {}
  };

  SmallVector<CIRArgs, 8> argInfo;

public:
  ClangToCIRArgMapping(const ASTContext &context, const CIRGenFunctionInfo &fi,
                       bool onlyRequiredArgs = false)
      : inallocaArgNo(invalidIndex), sRetArgNo(invalidIndex),
        argInfo(onlyRequiredArgs ? fi.getNumRequiredArgs() : fi.arg_size()) {
    construct(context, fi, onlyRequiredArgs);
  }

  bool hasSRetArg() const { return sRetArgNo != invalidIndex; }

  bool hasInallocaArg() const { return inallocaArgNo != invalidIndex; }

  unsigned totalCIRArgs() const { return totalCIRArguments; }

  bool hasPaddingArg(unsigned argNo) const {
    assert(argNo < argInfo.size());
    return argInfo[argNo].paddingArgIndex != invalidIndex;
  }

  /// Returns index of first CIR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getCIRArgs(unsigned argNo) const {
    assert(argNo < argInfo.size());
    return std::make_pair(argInfo[argNo].firstArgIndex,
                          argInfo[argNo].numberOfArgs);
  }

private:
  void construct(const ASTContext &context, const CIRGenFunctionInfo &fi,
                 bool onlyRequiredArgs);
};

void ClangToCIRArgMapping::construct(const ASTContext &context,
                                     const CIRGenFunctionInfo &fi,
                                     bool onlyRequiredArgs) {
  unsigned cirArgNo = 0;
  bool swapThisWithSRet = false;
  const ABIArgInfo &retAi = fi.getReturnInfo();

  assert(retAi.getKind() != ABIArgInfo::Indirect && "NYI");

  unsigned argNo = 0;
  unsigned numArgs = onlyRequiredArgs ? fi.getNumRequiredArgs() : fi.arg_size();
  for (CIRGenFunctionInfo::const_arg_iterator i = fi.arg_begin();
       argNo < numArgs; ++i, ++argNo) {
    assert(i != fi.arg_end());
    const ABIArgInfo &ai = i->info;
    // Collect data about CIR arguments corresponding to Clang argument ArgNo.
    auto &cirArgs = argInfo[argNo];

    assert(!ai.getPaddingType() && "NYI");

    switch (ai.getKind()) {
    default:
      llvm_unreachable("NYI");
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      // Postpone splitting structs into elements since this makes it way
      // more complicated for analysis to obtain information on the original
      // arguments.
      //
      // TODO(cir): a LLVM lowering prepare pass should break this down into
      // the appropriated pieces.
      assert(!MissingFeatures::constructABIArgDirectExtend());
      cirArgs.numberOfArgs = 1;
      break;
    }
    }

    if (cirArgs.numberOfArgs > 0) {
      cirArgs.firstArgIndex = cirArgNo;
      cirArgNo += cirArgs.numberOfArgs;
    }

    assert(!swapThisWithSRet && "NYI");
  }
  assert(argNo == argInfo.size());

  assert(!fi.usesInAlloca() && "NYI");

  totalCIRArguments = cirArgNo;
}

} // namespace

static bool hasInAllocaArgs(CIRGenModule &cgm, CallingConv explicitCc,
                            ArrayRef<QualType> argTypes) {
  assert(explicitCc != CC_Swift && explicitCc != CC_SwiftAsync && "Swift NYI");
  assert(!cgm.getTarget().getCXXABI().isMicrosoft() && "MSABI NYI");

  return false;
}

mlir::cir::FuncType CIRGenTypes::GetFunctionType(GlobalDecl gd) {
  const CIRGenFunctionInfo &fi = arrangeGlobalDeclaration(gd);
  return GetFunctionType(fi);
}

mlir::cir::FuncType CIRGenTypes::GetFunctionType(const CIRGenFunctionInfo &fi) {
  bool inserted = FunctionsBeingProcessed.insert(&fi).second;
  (void)inserted;
  assert(inserted && "Recursively being processed?");

  mlir::Type resultType = nullptr;
  const ABIArgInfo &retAI = fi.getReturnInfo();
  switch (retAI.getKind()) {
  case ABIArgInfo::Ignore:
    // TODO(CIR): This should probably be the None type from the builtin
    // dialect.
    resultType = nullptr;
    break;

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    resultType = retAI.getCoerceToType();
    break;

  default:
    assert(false && "NYI");
  }

  ClangToCIRArgMapping cirFunctionArgs(getContext(), fi, true);
  SmallVector<mlir::Type, 8> argTypes(cirFunctionArgs.totalCIRArgs());

  assert(!cirFunctionArgs.hasSRetArg() && "NYI");
  assert(!cirFunctionArgs.hasInallocaArg() && "NYI");

  // Add in all of the required arguments.
  unsigned argNo = 0;
  CIRGenFunctionInfo::const_arg_iterator it = fi.arg_begin(),
                                         ie = it + fi.getNumRequiredArgs();

  for (; it != ie; ++it, ++argNo) {
    const auto &argInfo = it->info;

    assert(!cirFunctionArgs.hasPaddingArg(argNo) && "NYI");

    unsigned firstCirArg, numCirArgs;
    std::tie(firstCirArg, numCirArgs) = cirFunctionArgs.getCIRArgs(argNo);

    switch (argInfo.getKind()) {
    default:
      llvm_unreachable("NYI");
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      mlir::Type argType = argInfo.getCoerceToType();
      // TODO: handle the test against llvm::StructType from codegen
      assert(numCirArgs == 1);
      argTypes[firstCirArg] = argType;
      break;
    }
    }
  }

  bool erased = FunctionsBeingProcessed.erase(&fi);
  (void)erased;
  assert(erased && "Not in set?");

  return mlir::cir::FuncType::get(
      argTypes, (resultType ? resultType : Builder.getVoidTy()),
      fi.isVariadic());
}

mlir::cir::FuncType CIRGenTypes::GetFunctionTypeForVTable(GlobalDecl gd) {
  const CXXMethodDecl *md = cast<CXXMethodDecl>(gd.getDecl());
  const FunctionProtoType *fpt = md->getType()->getAs<FunctionProtoType>();

  if (!isFuncTypeConvertible(fpt)) {
    llvm_unreachable("NYI");
    // return llvm::StructType::get(getLLVMContext());
  }

  return GetFunctionType(gd);
}

CIRGenCallee CIRGenCallee::prepareConcreteCallee(CIRGenFunction &cgf) const {
  if (isVirtual()) {
    const CallExpr *ce = getVirtualCallExpr();
    return cgf.cgm.getCXXABI().getVirtualFunctionPointer(
        cgf, getVirtualMethodDecl(), getThisAddress(), getVirtualFunctionType(),
        ce ? ce->getBeginLoc() : SourceLocation());
  }
  return *this;
}

void CIRGenFunction::buildAggregateStore(mlir::Value val, Address dest,
                                         bool destIsVolatile) {
  // In LLVM codegen:
  // Function to store a first-class aggregate into memory. We prefer to
  // store the elements rather than the aggregate to be more friendly to
  // fast-isel.
  // In CIR codegen:
  // Emit the most simple cir.store possible (e.g. a store for a whole
  // struct), which can later be broken down in other CIR levels (or prior
  // to dialect codegen).
  (void)destIsVolatile;
  // Stored result for the callers of this function expected to be in the same
  // scope as the value, don't make assumptions about current insertion point.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(val.getDefiningOp());
  builder.createStore(*currSrcLoc, val, dest);
}

static Address emitAddressAtOffset(CIRGenFunction &cgf, Address addr,
                                   const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    llvm_unreachable("NYI");
  }
  return addr;
}

static void addAttributesFromFunctionProtoType(CIRGenBuilderTy &builder,
                                               ASTContext &ctx,
                                               mlir::NamedAttrList &funcAttrs,
                                               const FunctionProtoType *fpt) {
  if (!fpt)
    return;

  if (!isUnresolvedExceptionSpec(fpt->getExceptionSpecType()) &&
      fpt->isNothrow()) {
    auto nu = mlir::cir::NoThrowAttr::get(builder.getContext());
    funcAttrs.set(nu.getMnemonic(), nu);
  }
}

/// Construct the CIR attribute list of a function or call.
///
/// When adding an attribute, please consider where it should be handled:
///
///   - getDefaultFunctionAttributes is for attributes that are essentially
///     part of the global target configuration (but perhaps can be
///     overridden on a per-function basis).  Adding attributes there
///     will cause them to also be set in frontends that build on Clang's
///     target-configuration logic, as well as for code defined in library
///     modules such as CUDA's libdevice.
///
///   - constructAttributeList builds on top of getDefaultFunctionAttributes
///     and adds declaration-specific, convention-specific, and
///     frontend-specific logic.  The last is of particular importance:
///     attributes that restrict how the frontend generates code must be
///     added here rather than getDefaultFunctionAttributes.
///
void CIRGenModule::constructAttributeList(StringRef name,
                                          const CIRGenFunctionInfo &fi,
                                          CIRGenCalleeInfo calleeInfo,
                                          mlir::NamedAttrList &funcAttrs,
                                          mlir::cir::CallingConv &callingConv,
                                          bool attrOnCallSite, bool isThunk) {
  // Implementation Disclaimer
  //
  // UnimplementedFeature and asserts are used throughout the code to track
  // unsupported and things not yet implemented. However, most of the content of
  // this function is on detecting attributes, which doesn't not cope with
  // existing approaches to track work because its too big.
  //
  // That said, for the most part, the approach here is very specific compared
  // to the rest of CIRGen and attributes and other handling should be done upon
  // demand.

  // Collect function CIR attributes from the CC lowering.
  callingConv = fi.getEffectiveCallingConvention();
  // TODO: NoReturn, cmse_nonsecure_call

  // Collect function CIR attributes from the callee prototype if we have one.
  addAttributesFromFunctionProtoType(getBuilder(), astCtx, funcAttrs,
                                     calleeInfo.getCalleeFunctionProtoType());

  const Decl *targetDecl = calleeInfo.getCalleeDecl().getDecl();

  // TODO(cir): Attach assumption attributes to the declaration. If this is a
  // call site, attach assumptions from the caller to the call as well.

  bool hasOptnone = false;
  (void)hasOptnone;
  // The NoBuiltinAttr attached to the target FunctionDecl.
  mlir::Attribute *nba;

  if (targetDecl) {

    if (targetDecl->hasAttr<NoThrowAttr>()) {
      auto nu = mlir::cir::NoThrowAttr::get(builder.getContext());
      funcAttrs.set(nu.getMnemonic(), nu);
    }

    if (const FunctionDecl *fn = dyn_cast<FunctionDecl>(targetDecl)) {
      addAttributesFromFunctionProtoType(
          getBuilder(), astCtx, funcAttrs,
          fn->getType()->getAs<FunctionProtoType>());
      if (attrOnCallSite && fn->isReplaceableGlobalAllocationFunction()) {
        // A sane operator new returns a non-aliasing pointer.
        auto kind = fn->getDeclName().getCXXOverloadedOperator();
        if (getCodeGenOpts().AssumeSaneOperatorNew &&
            (kind == OO_New || kind == OO_Array_New))
          // TODO(CIR): attr
          ; // llvm::Attribute::NoAlias
      }
      const CXXMethodDecl *md = dyn_cast<CXXMethodDecl>(fn);
      const bool isVirtualCall = md && md->isVirtual();
      // Don't use [[noreturn]], _Noreturn or [[no_builtin]] for a call to a
      // virtual function. These attributes are not inherited by overloads.
      if (!(attrOnCallSite && isVirtualCall)) {
        if (fn->isNoReturn())
          ; // NoReturn
        // NBA = Fn->getAttr<NoBuiltinAttr>();
        // TODO(CIR): attr
        (void)nba;
      }
    }

    if (isa<FunctionDecl>(targetDecl) || isa<VarDecl>(targetDecl)) {
      // Only place nomerge attribute on call sites, never functions. This
      // allows it to work on indirect virtual function calls.
      if (attrOnCallSite && targetDecl->hasAttr<NoMergeAttr>())
        ; // TODO(CIR): attr
    }

    // 'const', 'pure' and 'noalias' attributed functions are also nounwind.
    if (targetDecl->hasAttr<ConstAttr>()) {
      // gcc specifies that 'const' functions have greater restrictions than
      // 'pure' functions, so they also cannot have infinite loops.
      // TODO(CIR): attr
    } else if (targetDecl->hasAttr<PureAttr>()) {
      // gcc specifies that 'pure' functions cannot have infinite loops.
      // TODO(CIR): attr
    } else if (targetDecl->hasAttr<NoAliasAttr>()) {
      // TODO(CIR): attr
    }

    hasOptnone = targetDecl->hasAttr<OptimizeNoneAttr>();
    if (auto *allocSize = targetDecl->getAttr<AllocSizeAttr>()) {
      std::optional<unsigned> numElemsParam;
      if (allocSize->getNumElemsParam().isValid())
        numElemsParam = allocSize->getNumElemsParam().getLLVMIndex();
      // TODO(cir): add alloc size attr.
    }

    if (targetDecl->hasAttr<OpenCLKernelAttr>()) {
      auto cirKernelAttr =
          mlir::cir::OpenCLKernelAttr::get(builder.getContext());
      funcAttrs.set(cirKernelAttr.getMnemonic(), cirKernelAttr);

      auto uniformAttr = mlir::cir::OpenCLKernelUniformWorkGroupSizeAttr::get(
          builder.getContext());
      if (getLangOpts().OpenCLVersion <= 120) {
        // OpenCL v1.2 Work groups are always uniform
        funcAttrs.set(uniformAttr.getMnemonic(), uniformAttr);
      } else {
        // OpenCL v2.0 Work groups may be whether uniform or not.
        // '-cl-uniform-work-group-size' compile option gets a hint
        // to the compiler that the global work-size be a multiple of
        // the work-group size specified to clEnqueueNDRangeKernel
        // (i.e. work groups are uniform).
        if (getLangOpts().OffloadUniformBlock) {
          funcAttrs.set(uniformAttr.getMnemonic(), uniformAttr);
        }
      }
    }

    if (targetDecl->hasAttr<CUDAGlobalAttr>() &&
        getLangOpts().OffloadUniformBlock)
      assert(!MissingFeatures::cuda());

    if (targetDecl->hasAttr<ArmLocallyStreamingAttr>())
      ; // TODO(CIR): attr
  }

  getDefaultFunctionAttributes(name, hasOptnone, attrOnCallSite, funcAttrs);
}

static mlir::cir::CIRCallOpInterface
buildCallLikeOp(CIRGenFunction &cgf, mlir::Location callLoc,
                mlir::cir::FuncType indirectFuncTy, mlir::Value indirectFuncVal,
                mlir::cir::FuncOp directFuncOp,
                SmallVectorImpl<mlir::Value> &cirCallArgs, bool isInvoke,
                mlir::cir::CallingConv callingConv,
                mlir::cir::ExtraFuncAttributesAttr extraFnAttrs) {
  auto &builder = cgf.getBuilder();
  auto getOrCreateSurroundingTryOp = [&]() {
    // In OG, we build the landing pad for this scope. In CIR, we emit a
    // synthetic cir.try because this didn't come from codegenerating from a
    // try/catch in C++.
    assert(cgf.currLexScope && "expected scope");
    mlir::cir::TryOp op = cgf.currLexScope->getClosestTryParent();
    if (op)
      return op;

    op = builder.create<mlir::cir::TryOp>(
        *cgf.currSrcLoc, /*scopeBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {},
        // Don't emit the code right away for catch clauses, for
        // now create the regions and consume the try scope result.
        // Note that clauses are later populated in
        // CIRGenFunction::buildLandingPad.
        [&](mlir::OpBuilder &b, mlir::Location loc,
            mlir::OperationState &result) {
          // Since this didn't come from an explicit try, we only need one
          // handler: unwind.
          auto *r = result.addRegion();
          builder.createBlock(r);
        });
    op.setSynthetic(true);
    return op;
  };

  if (isInvoke) {
    // This call can throw, few options:
    //  - If this call does not have an associated cir.try, use the
    //    one provided by InvokeDest,
    //  - User written try/catch clauses require calls to handle
    //    exceptions under cir.try.
    auto tryOp = getOrCreateSurroundingTryOp();
    assert(tryOp && "expected");

    mlir::OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
    if (tryOp.getSynthetic()) {
      mlir::Block *lastBlock = &tryOp.getTryRegion().back();
      builder.setInsertionPointToStart(lastBlock);
    } else {
      assert(builder.getInsertionBlock() && "expected valid basic block");
    }

    mlir::cir::CallOp callOpWithExceptions;
    // TODO(cir): Set calling convention for `cir.try_call`.
    assert(callingConv == mlir::cir::CallingConv::C && "NYI");
    if (indirectFuncTy) {
      callOpWithExceptions = builder.createIndirectTryCallOp(
          callLoc, indirectFuncVal, indirectFuncTy, cirCallArgs);
    } else {
      callOpWithExceptions =
          builder.createTryCallOp(callLoc, directFuncOp, cirCallArgs);
    }
    callOpWithExceptions->setAttr("extra_attrs", extraFnAttrs);

    cgf.callWithExceptionCtx = callOpWithExceptions;
    auto *invokeDest = cgf.getInvokeDest(tryOp);
    (void)invokeDest;
    cgf.callWithExceptionCtx = nullptr;

    if (tryOp.getSynthetic()) {
      builder.create<mlir::cir::YieldOp>(tryOp.getLoc());
      builder.restoreInsertionPoint(ip);
    }
    return callOpWithExceptions;
  }

  assert(builder.getInsertionBlock() && "expected valid basic block");
  if (indirectFuncTy) {
    // TODO(cir): Set calling convention for indirect calls.
    assert(callingConv == mlir::cir::CallingConv::C && "NYI");
    return builder.createIndirectCallOp(
        callLoc, indirectFuncVal, indirectFuncTy, cirCallArgs,
        mlir::cir::CallingConv::C, extraFnAttrs);
  }
  return builder.createCallOp(callLoc, directFuncOp, cirCallArgs, callingConv,
                              extraFnAttrs);
}

RValue CIRGenFunction::buildCall(const CIRGenFunctionInfo &callInfo,
                                 const CIRGenCallee &callee,
                                 ReturnValueSlot returnValue,
                                 const CallArgList &callArgs,
                                 mlir::cir::CIRCallOpInterface *callOrTryCall,
                                 bool isMustTail, mlir::Location loc,
                                 std::optional<const clang::CallExpr *> e) {
  auto builder = cgm.getBuilder();
  // FIXME: We no longer need the types from CallArgs; lift up and simplify

  assert(callee.isOrdinary() || callee.isVirtual());

  // Handle struct-return functions by passing a pointer to the location that we
  // would like to return info.
  QualType retTy = callInfo.getReturnType();
  const auto &retAi = callInfo.getReturnInfo();

  mlir::cir::FuncType cirFuncTy = getTypes().GetFunctionType(callInfo);

  const Decl *targetDecl = callee.getAbstractInfo().getCalleeDecl().getDecl();
  // This is not always tied to a FunctionDecl (e.g. builtins that are xformed
  // into calls to other functions)
  if (const FunctionDecl *fd = dyn_cast_or_null<FunctionDecl>(targetDecl)) {
    // We can only guarantee that a function is called from the correct
    // context/function based on the appropriate target attributes,
    // so only check in the case where we have both always_inline and target
    // since otherwise we could be making a conditional call after a check for
    // the proper cpu features (and it won't cause code generation issues due to
    // function based code generation).
    if (targetDecl->hasAttr<AlwaysInlineAttr>() &&
        (targetDecl->hasAttr<TargetAttr>() ||
         (CurFuncDecl && CurFuncDecl->hasAttr<TargetAttr>()))) {
      // FIXME(cir): somehow refactor this function to use SourceLocation?
      SourceLocation loc;
      checkTargetFeatures(loc, fd);
    }

    // Some architectures (such as x86-64) have the ABI changed based on
    // attribute-target/features. Give them a chance to diagnose.
    assert(!MissingFeatures::checkFunctionCallABI());
  }

  // TODO: add DNEBUG code

  // 1. Set up the arguments

  // If we're using inalloca, insert the allocation after the stack save.
  // FIXME: Do this earlier rather than hacking it in here!
  Address argMemory = Address::invalid();
  assert(!callInfo.getArgStruct() && "NYI");

  ClangToCIRArgMapping cirFunctionArgs(cgm.getASTContext(), callInfo);
  SmallVector<mlir::Value, 16> cirCallArgs(cirFunctionArgs.totalCIRArgs());

  // If the call returns a temporary with struct return, create a temporary
  // alloca to hold the result, unless one is given to us.
  assert(!retAi.isIndirect() && !retAi.isInAlloca() &&
         !retAi.isCoerceAndExpand() && "NYI");

  // When passing arguments using temporary allocas, we need to add the
  // appropriate lifetime markers. This vector keeps track of all the lifetime
  // markers that need to be ended right after the call.
  assert(!MissingFeatures::shouldEmitLifetimeMarkers() && "NYI");

  // Translate all of the arguments as necessary to match the CIR lowering.
  assert(callInfo.arg_size() == callArgs.size() &&
         "Mismatch between function signature & arguments.");
  unsigned argNo = 0;
  CIRGenFunctionInfo::const_arg_iterator infoIt = callInfo.arg_begin();
  for (CallArgList::const_iterator i = callArgs.begin(), e = callArgs.end();
       i != e; ++i, ++infoIt, ++argNo) {
    const ABIArgInfo &argInfo = infoIt->info;

    // Insert a padding argument to ensure proper alignment.
    assert(!cirFunctionArgs.hasPaddingArg(argNo) && "Padding args NYI");

    unsigned firstCirArg, numCirArgs;
    std::tie(firstCirArg, numCirArgs) = cirFunctionArgs.getCIRArgs(argNo);

    switch (argInfo.getKind()) {
    case ABIArgInfo::Direct: {
      if (!mlir::isa<mlir::cir::StructType>(argInfo.getCoerceToType()) &&
          argInfo.getCoerceToType() == convertType(infoIt->type) &&
          argInfo.getDirectOffset() == 0) {
        assert(numCirArgs == 1);
        mlir::Value v;
        assert(!i->isAggregate() && "Aggregate NYI");
        v = i->getKnownRValue().getScalarVal();

        assert(callInfo.getExtParameterInfo(argNo).getABI() !=
                   ParameterABI::SwiftErrorResult &&
               "swift NYI");

        // We might have to widen integers, but we should never truncate.
        if (argInfo.getCoerceToType() != v.getType() &&
            mlir::isa<mlir::cir::IntType>(v.getType()))
          llvm_unreachable("NYI");

        // If the argument doesn't match, perform a bitcast to coerce it. This
        // can happen due to trivial type mismatches.
        if (firstCirArg < cirFuncTy.getNumInputs() &&
            v.getType() != cirFuncTy.getInput(firstCirArg))
          v = builder.createBitcast(v, cirFuncTy.getInput(firstCirArg));

        cirCallArgs[firstCirArg] = v;
        break;
      }

      // FIXME: Avoid the conversion through memory if possible.
      Address src = Address::invalid();
      if (!i->isAggregate()) {
        llvm_unreachable("NYI");
      } else {
        src = i->hasLValue() ? i->getKnownLValue().getAddress()
                             : i->getKnownRValue().getAggregateAddress();
      }

      // If the value is offset in memory, apply the offset now.
      src = emitAddressAtOffset(*this, src, argInfo);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      auto sTy = dyn_cast<mlir::cir::StructType>(argInfo.getCoerceToType());
      if (sTy && argInfo.isDirect() && argInfo.getCanBeFlattened()) {
        auto srcTy = src.getElementType();
        // FIXME(cir): get proper location for each argument.
        auto argLoc = loc;

        // If the source type is smaller than the destination type of the
        // coerce-to logic, copy the source value into a temp alloca the size
        // of the destination type to allow loading all of it. The bits past
        // the source value are left undef.
        // FIXME(cir): add data layout info and compare sizes instead of
        // matching the types.
        //
        // uint64_t SrcSize = CGM.getDataLayout().getTypeAllocSize(SrcTy);
        // uint64_t DstSize = CGM.getDataLayout().getTypeAllocSize(STy);
        // if (SrcSize < DstSize) {
        if (srcTy != sTy)
          llvm_unreachable("NYI");
        else {
          // FIXME(cir): this currently only runs when the types are different,
          // but should be when alloc sizes are different, fix this as soon as
          // datalayout gets introduced.
          src = builder.createElementBitCast(argLoc, src, sTy);
        }

        // assert(NumCIRArgs == STy.getMembers().size());
        // In LLVMGen: Still only pass the struct without any gaps but mark it
        // as such somehow.
        //
        // In CIRGen: Emit a load from the "whole" struct,
        // which shall be broken later by some lowering step into multiple
        // loads.
        assert(numCirArgs == 1 && "dont break up arguments here!");
        cirCallArgs[firstCirArg] = builder.createLoad(argLoc, src);
      } else {
        llvm_unreachable("NYI");
      }

      break;
    }
    default:
      assert(false && "Only Direct support so far");
    }
  }

  const CIRGenCallee &concreteCallee = callee.prepareConcreteCallee(*this);
  auto *calleePtr = concreteCallee.getFunctionPointer();

  // If we're using inalloca, set up that argument.
  assert(!argMemory.isValid() && "inalloca NYI");

  // 2. Prepare the function pointer.

  // TODO: simplifyVariadicCallee

  // 3. Perform the actual call.

  // TODO: Deactivate any cleanups that we're supposed to do immediately before
  // the call.
  // if (!CallArgs.getCleanupsToDeactivate().empty())
  //   deactivateArgCleanupsBeforeCall(*this, CallArgs);
  // TODO: Update the largest vector width if any arguments have vector types.

  // Compute the calling convention and attributes.
  mlir::NamedAttrList attrs;
  StringRef fnName;
  if (auto calleeFnOp = dyn_cast<mlir::cir::FuncOp>(calleePtr))
    fnName = calleeFnOp.getName();

  mlir::cir::CallingConv callingConv;
  cgm.constructAttributeList(fnName, callInfo, callee.getAbstractInfo(), attrs,
                             callingConv,
                             /*AttrOnCallSite=*/true,
                             /*IsThunk=*/false);

  // TODO: strictfp
  // TODO: Add call-site nomerge, noinline, always_inline attribute if exists.

  // Apply some call-site-specific attributes.
  // TODO: work this into building the attribute set.

  // Apply always_inline to all calls within flatten functions.
  // FIXME: should this really take priority over __try, below?
  // assert(!CurCodeDecl->hasAttr<FlattenAttr>() &&
  //        !TargetDecl->hasAttr<NoInlineAttr>() && "NYI");

  // Disable inlining inside SEH __try blocks.
  if (isSEHTryScope())
    llvm_unreachable("NYI");

  // Decide whether to use a call or an invoke.
  bool cannotThrow;
  if (currentFunctionUsesSEHTry()) {
    // SEH cares about asynchronous exceptions, so everything can "throw."
    cannotThrow = false;
  } else if (isCleanupPadScope() &&
             EHPersonality::get(*this).isMSVCXXPersonality()) {
    // The MSVC++ personality will implicitly terminate the program if an
    // exception is thrown during a cleanup outside of a try/catch.
    // We don't need to model anything in IR to get this behavior.
    cannotThrow = true;
  } else {
    // Otherwise, nounwind call sites will never throw.
    auto noThrowAttr = mlir::cir::NoThrowAttr::get(builder.getContext());
    cannotThrow = attrs.getNamed(noThrowAttr.getMnemonic()).has_value();

    if (auto fptr = dyn_cast<mlir::cir::FuncOp>(calleePtr))
      if (fptr.getExtraAttrs().getElements().contains(
              noThrowAttr.getMnemonic()))
        cannotThrow = true;
  }
  bool isInvoke = cannotThrow ? false : isInvokeDest();

  // TODO: UnusedReturnSizePtr
  if (const FunctionDecl *fd = dyn_cast_or_null<FunctionDecl>(CurFuncDecl))
    assert(!fd->hasAttr<StrictFPAttr>() && "NYI");

  // TODO: alignment attributes

  auto callLoc = loc;
  mlir::cir::CIRCallOpInterface theCall = [&]() {
    mlir::cir::FuncType indirectFuncTy;
    mlir::Value indirectFuncVal;
    mlir::cir::FuncOp directFuncOp;

    if (auto fnOp = dyn_cast<mlir::cir::FuncOp>(calleePtr)) {
      directFuncOp = fnOp;
    } else if (auto getGlobalOp = dyn_cast<mlir::cir::GetGlobalOp>(calleePtr)) {
      // FIXME(cir): This peephole optimization to avoids indirect calls for
      // builtins. This should be fixed in the builting declaration instead by
      // not emitting an unecessary get_global in the first place.
      auto *globalOp = mlir::SymbolTable::lookupSymbolIn(cgm.getModule(),
                                                         getGlobalOp.getName());
      assert(getGlobalOp && "undefined global function");
      directFuncOp = llvm::dyn_cast<mlir::cir::FuncOp>(globalOp);
      assert(directFuncOp && "operation is not a function");
    } else {
      [[maybe_unused]] auto resultTypes = calleePtr->getResultTypes();
      [[maybe_unused]] auto funcPtrTy =
          mlir::dyn_cast<mlir::cir::PointerType>(resultTypes.front());
      assert(funcPtrTy &&
             mlir::isa<mlir::cir::FuncType>(funcPtrTy.getPointee()) &&
             "expected pointer to function");

      indirectFuncTy = cirFuncTy;
      indirectFuncVal = calleePtr->getResult(0);
    }

    auto extraFnAttrs = mlir::cir::ExtraFuncAttributesAttr::get(
        builder.getContext(), attrs.getDictionary(builder.getContext()));

    mlir::cir::CIRCallOpInterface callLikeOp = buildCallLikeOp(
        *this, callLoc, indirectFuncTy, indirectFuncVal, directFuncOp,
        cirCallArgs, isInvoke, callingConv, extraFnAttrs);

    if (e)
      callLikeOp->setAttr(
          "ast", mlir::cir::ASTCallExprAttr::get(builder.getContext(), *e));

    if (callOrTryCall)
      *callOrTryCall = callLikeOp;
    return callLikeOp;
  }();

  if (const auto *fd = dyn_cast_or_null<FunctionDecl>(CurFuncDecl))
    assert(!fd->getAttr<CFGuardAttr>() && "NYI");

  // TODO: set attributes on callop
  // assert(!theCall.getResults().getType().front().isSignlessInteger() &&
  //        "Vector NYI");
  // TODO: LLVM models indirect calls via a null callee, how should we do this?
  assert(!cgm.getLangOpts().ObjCAutoRefCount && "Not supported");
  assert((!targetDecl || !targetDecl->hasAttr<NotTailCalledAttr>()) && "NYI");
  assert(!getDebugInfo() && "No debug info yet");
  assert((!targetDecl || !targetDecl->hasAttr<ErrorAttr>()) && "NYI");

  // 4. Finish the call.

  // If the call doesn't return, finish the basic block and clear the insertion
  // point; this allows the rest of CIRGen to discard unreachable code.
  // TODO: figure out how to support doesNotReturn

  assert(!isMustTail && "NYI");

  // TODO: figure out writebacks? seems like ObjC only __autorelease

  // TODO: cleanup argument memory at the end

  // Extract the return value.
  RValue ret = [&] {
    switch (retAi.getKind()) {
    case ABIArgInfo::Direct: {
      mlir::Type retCirTy = convertType(retTy);
      if (retAi.getCoerceToType() == retCirTy && retAi.getDirectOffset() == 0) {
        switch (getEvaluationKind(retTy)) {
        case TEK_Aggregate: {
          Address destPtr = returnValue.getValue();
          bool destIsVolatile = returnValue.isVolatile();

          if (!destPtr.isValid()) {
            destPtr = CreateMemTemp(retTy, callLoc, getCounterAggTmpAsString());
            destIsVolatile = false;
          }

          auto results = theCall->getOpResults();
          assert(results.size() <= 1 && "multiple returns NYI");

          SourceLocRAIIObject loc{*this, callLoc};
          buildAggregateStore(results[0], destPtr, destIsVolatile);
          return RValue::getAggregate(destPtr);
        }
        case TEK_Scalar: {
          // If the argument doesn't match, perform a bitcast to coerce it. This
          // can happen due to trivial type mismatches.
          auto results = theCall->getOpResults();
          assert(results.size() <= 1 && "multiple returns NYI");
          assert(results[0].getType() == retCirTy && "Bitcast support NYI");
          return RValue::get(results[0]);
        }
        default:
          llvm_unreachable("NYI");
        }
      } else {
        llvm_unreachable("No other forms implemented yet.");
      }
    }

    case ABIArgInfo::Ignore:
      // If we are ignoring an argument that had a result, make sure to
      // construct the appropriate return value for our caller.
      return GetUndefRValue(retTy);

    default:
      llvm_unreachable("NYI");
    }

    llvm_unreachable("NYI");
    return RValue{};
  }();

  // TODO: implement assumed_aligned

  // TODO: implement lifetime extensions

  assert(retTy.isDestructedType() != QualType::DK_nontrivial_c_struct && "NYI");

  return ret;
}

mlir::Value CIRGenFunction::buildRuntimeCall(mlir::Location loc,
                                             mlir::cir::FuncOp callee,
                                             ArrayRef<mlir::Value> args) {
  // TODO(cir): set the calling convention to this runtime call.
  assert(!MissingFeatures::setCallingConv());

  auto call = builder.createCallOp(loc, callee, args);
  assert(call->getNumResults() <= 1 &&
         "runtime functions have at most 1 result");

  if (call->getNumResults() == 0)
    return nullptr;

  return call->getResult(0);
}

void CIRGenFunction::buildCallArg(CallArgList &args, const Expr *e,
                                  QualType type) {
  // TODO: Add the DisableDebugLocationUpdates helper
  assert(!dyn_cast<ObjCIndirectCopyRestoreExpr>(e) && "NYI");

  assert(type->isReferenceType() == e->isGLValue() &&
         "reference binding to unmaterialized r-value!");

  if (e->isGLValue()) {
    assert(e->getObjectKind() == OK_Ordinary);
    return args.add(buildReferenceBindingToExpr(e), type);
  }

  bool hasAggregateEvalKind = hasAggregateEvaluationKind(type);

  // In the Microsoft C++ ABI, aggregate arguments are destructed by the callee.
  // However, we still have to push an EH-only cleanup in case we unwind before
  // we make it to the call.
  if (type->isRecordType() &&
      type->castAs<RecordType>()->getDecl()->isParamDestroyedInCallee()) {
    llvm_unreachable("Microsoft C++ ABI is NYI");
  }

  if (hasAggregateEvalKind && isa<ImplicitCastExpr>(e) &&
      cast<CastExpr>(e)->getCastKind() == CK_LValueToRValue) {
    LValue l = buildLValue(cast<CastExpr>(e)->getSubExpr());
    assert(l.isSimple());
    args.addUncopiedAggregate(l, type);
    return;
  }

  args.add(buildAnyExprToTemp(e), type);
}

QualType CIRGenFunction::getVarArgType(const Expr *arg) {
  // System headers on Windows define NULL to 0 instead of 0LL on Win64. MSVC
  // implicitly widens null pointer constants that are arguments to varargs
  // functions to pointer-sized ints.
  if (!getTarget().getTriple().isOSWindows())
    return arg->getType();

  if (arg->getType()->isIntegerType() &&
      getContext().getTypeSize(arg->getType()) <
          getContext().getTargetInfo().getPointerWidth(LangAS::Default) &&
      arg->isNullPointerConstant(getContext(),
                                 Expr::NPC_ValueDependentIsNotNull)) {
    return getContext().getIntPtrType();
  }

  return arg->getType();
}

/// Similar to buildAnyExpr(), however, the result will always be accessible
/// even if no aggregate location is provided.
RValue CIRGenFunction::buildAnyExprToTemp(const Expr *e) {
  AggValueSlot aggSlot = AggValueSlot::ignored();

  if (hasAggregateEvaluationKind(e->getType()))
    aggSlot = CreateAggTemp(e->getType(), getLoc(e->getSourceRange()),
                            getCounterAggTmpAsString());

  return buildAnyExpr(e, aggSlot);
}

void CIRGenFunction::buildCallArgs(
    CallArgList &args, PrototypeWrapper prototype,
    llvm::iterator_range<CallExpr::const_arg_iterator> argRange,
    AbstractCallee ac, unsigned paramsToSkip, EvaluationOrder order) {

  llvm::SmallVector<QualType, 16> argTypes;

  assert((paramsToSkip == 0 || prototype.P) &&
         "Can't skip parameters if type info is not provided");

  // This variable only captures *explicitly* written conventions, not those
  // applied by default via command line flags or target defaults, such as
  // thiscall, appcs, stdcall via -mrtd, etc. Computing that correctly would
  // require knowing if this is a C++ instance method or being able to see
  // unprotyped FunctionTypes.
  CallingConv explicitCc = CC_C;

  // First, if a prototype was provided, use those argument types.
  bool isVariadic = false;
  if (prototype.P) {
    const auto *md = mlir::dyn_cast<const ObjCMethodDecl *>(prototype.P);
    assert(!md && "ObjCMethodDecl NYI");

    const auto *fpt = prototype.P.get<const FunctionProtoType *>();
    isVariadic = fpt->isVariadic();
    explicitCc = fpt->getExtInfo().getCC();
    argTypes.assign(fpt->param_type_begin() + paramsToSkip,
                    fpt->param_type_end());
  }

  // If we still have any arguments, emit them using the type of the argument.
  for (auto *a : llvm::drop_begin(argRange, argTypes.size()))
    argTypes.push_back(isVariadic ? getVarArgType(a) : a->getType());
  assert((int)argTypes.size() == (argRange.end() - argRange.begin()));

  // We must evaluate arguments from right to left in the MS C++ ABI, because
  // arguments are destroyed left to right in the callee. As a special case,
  // there are certain language constructs taht require left-to-right
  // evaluation, and in those cases we consider the evaluation order requirement
  // to trump the "destruction order is reverse construction order" guarantee.
  bool leftToRight = true;
  assert(!cgm.getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee() &&
         "MSABI NYI");
  assert(!hasInAllocaArgs(cgm, explicitCc, argTypes) && "NYI");

  auto maybeEmitImplicitObjectSize = [&](unsigned i, const Expr *arg,
                                         RValue emittedArg) {
    if (!ac.hasFunctionDecl() || i >= ac.getNumParams())
      return;
    auto *ps = ac.getParamDecl(i)->getAttr<PassObjectSizeAttr>();
    if (ps == nullptr)
      return;

    const auto &context = getContext();
    auto sizeTy = context.getSizeType();
    auto t = builder.getUIntNTy(context.getTypeSize(sizeTy));
    assert(emittedArg.getScalarVal() && "We emitted nothing for the arg?");
    auto v = evaluateOrEmitBuiltinObjectSize(
        arg, ps->getType(), t, emittedArg.getScalarVal(), ps->isDynamic());
    args.add(RValue::get(v), sizeTy);
    // If we're emitting args in reverse, be sure to do so with
    // pass_object_size, as well.
    if (!leftToRight)
      std::swap(args.back(), *(&args.back() - 1));
  };

  // Evaluate each argument in the appropriate order.
  size_t callArgsStart = args.size();
  for (unsigned i = 0, e = argTypes.size(); i != e; ++i) {
    unsigned idx = leftToRight ? i : e - i - 1;
    CallExpr::const_arg_iterator arg = argRange.begin() + idx;
    unsigned initialArgSize = args.size();
    assert(!isa<ObjCIndirectCopyRestoreExpr>(*arg) && "NYI");
    assert(!isa_and_nonnull<ObjCMethodDecl>(ac.getDecl()) && "NYI");

    buildCallArg(args, *arg, argTypes[idx]);
    // In particular, we depend on it being the last arg in Args, and the
    // objectsize bits depend on there only being one arg if !LeftToRight.
    assert(initialArgSize + 1 == args.size() &&
           "The code below depends on only adding one arg per buildCallArg");
    (void)initialArgSize;
    // Since pointer argument are never emitted as LValue, it is safe to emit
    // non-null argument check for r-value only.
    if (!args.back().hasLValue()) {
      RValue rvArg = args.back().getKnownRValue();
      assert(!SanOpts.has(SanitizerKind::NonnullAttribute) && "Sanitizers NYI");
      assert(!SanOpts.has(SanitizerKind::NullabilityArg) && "Sanitizers NYI");
      // @llvm.objectsize should never have side-effects and shouldn't need
      // destruction/cleanups, so we can safely "emit" it after its arg,
      // regardless of right-to-leftness
      maybeEmitImplicitObjectSize(idx, *arg, rvArg);
    }
  }

  if (!leftToRight) {
    // Un-reverse the arguments we just evaluated so they match up with the CIR
    // function.
    std::reverse(args.begin() + callArgsStart, args.end());
  }
}

/// Returns the canonical formal type of the given C++ method.
static CanQual<FunctionProtoType> getFormalType(const CXXMethodDecl *md) {
  return md->getType()
      ->getCanonicalTypeUnqualified()
      .getAs<FunctionProtoType>();
}

/// TODO(cir): this should be shared with LLVM codegen
static void addExtParameterInfosForCall(
    llvm::SmallVectorImpl<FunctionProtoType::ExtParameterInfo> &paramInfos,
    const FunctionProtoType *proto, unsigned prefixArgs, unsigned totalArgs) {
  assert(proto->hasExtParameterInfos());
  assert(paramInfos.size() <= prefixArgs);
  assert(proto->getNumParams() + prefixArgs <= totalArgs);

  paramInfos.reserve(totalArgs);

  // Add default infos for any prefix args that don't already have infos.
  paramInfos.resize(prefixArgs);

  // Add infos for the prototype.
  for (const auto &paramInfo : proto->getExtParameterInfos()) {
    paramInfos.push_back(paramInfo);
    // pass_object_size params have no parameter info.
    if (paramInfo.hasPassObjectSize())
      paramInfos.emplace_back();
  }

  assert(paramInfos.size() <= totalArgs &&
         "Did we forget to insert pass_object_size args?");
  // Add default infos for the variadic and/or suffix arguments.
  paramInfos.resize(totalArgs);
}

/// Adds the formal parameters in FPT to the given prefix. If any parameter in
/// FPT has pass_object_size_attrs, then we'll add parameters for those, too.
/// TODO(cir): this should be shared with LLVM codegen
static void appendParameterTypes(
    const CIRGenTypes &cgt, SmallVectorImpl<CanQualType> &prefix,
    SmallVectorImpl<FunctionProtoType::ExtParameterInfo> &paramInfos,
    CanQual<FunctionProtoType> fpt) {
  // Fast path: don't touch param info if we don't need to.
  if (!fpt->hasExtParameterInfos()) {
    assert(paramInfos.empty() &&
           "We have paramInfos, but the prototype doesn't?");
    prefix.append(fpt->param_type_begin(), fpt->param_type_end());
    return;
  }

  unsigned prefixSize = prefix.size();
  // In the vast majority of cases, we'll have precisely FPT->getNumParams()
  // parameters; the only thing that can change this is the presence of
  // pass_object_size. So, we preallocate for the common case.
  prefix.reserve(prefix.size() + fpt->getNumParams());

  auto extInfos = fpt->getExtParameterInfos();
  assert(extInfos.size() == fpt->getNumParams());
  for (unsigned i = 0, e = fpt->getNumParams(); i != e; ++i) {
    prefix.push_back(fpt->getParamType(i));
    if (extInfos[i].hasPassObjectSize())
      prefix.push_back(cgt.getContext().getSizeType());
  }

  addExtParameterInfosForCall(paramInfos, fpt.getTypePtr(), prefixSize,
                              prefix.size());
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXStructorDeclaration(GlobalDecl gd) {
  auto *md = cast<CXXMethodDecl>(gd.getDecl());

  llvm::SmallVector<CanQualType, 16> argTypes;
  SmallVector<FunctionProtoType::ExtParameterInfo, 16> paramInfos;
  argTypes.push_back(DeriveThisType(md->getParent(), md));

  bool passParams = true;

  if (auto *cd = dyn_cast<CXXConstructorDecl>(md)) {
    // A base class inheriting constructor doesn't get forwarded arguments
    // needed to construct a virtual base (or base class thereof)
    assert(!cd->getInheritedConstructor() && "Inheritance NYI");
  }

  CanQual<FunctionProtoType> ftp = getFormalType(md);

  if (passParams)
    appendParameterTypes(*this, argTypes, paramInfos, ftp);

  assert(paramInfos.empty() && "NYI");

  assert(!md->isVariadic() && "Variadic fns NYI");
  RequiredArgs required = RequiredArgs::All;
  (void)required;

  FunctionType::ExtInfo extInfo = ftp->getExtInfo();

  assert(!TheCXXABI.HasThisReturn(gd) && "NYI");

  CanQualType resultType = Context.VoidTy;
  (void)resultType;

  return arrangeCIRFunctionInfo(resultType, FnInfoOpts::IsInstanceMethod,
                                argTypes, extInfo, paramInfos, required);
}

/// Derives the 'this' type for CIRGen purposes, i.e. ignoring method CVR
/// qualification. Either or both of RD and MD may be null. A null RD indicates
/// that there is no meaningful 'this' type, and a null MD can occur when
/// calling a method pointer.
CanQualType CIRGenTypes::DeriveThisType(const CXXRecordDecl *rd,
                                        const CXXMethodDecl *md) {
  QualType recTy;
  if (rd)
    recTy = getContext().getTagDeclType(rd)->getCanonicalTypeInternal();
  else
    assert(false && "CXXMethodDecl NYI");

  if (md)
    recTy = getContext().getAddrSpaceQualType(
        recTy, md->getMethodQualifiers().getAddressSpace());
  return getContext().getPointerType(CanQualType::CreateUnsafe(recTy));
}

/// Arrange the CIR function layout for a value of the given function type, on
/// top of any implicit parameters already stored.
static const CIRGenFunctionInfo &
arrangeCIRFunctionInfo(CIRGenTypes &cgt, FnInfoOpts instanceMethod,
                       SmallVectorImpl<CanQualType> &prefix,
                       CanQual<FunctionProtoType> ftp) {
  SmallVector<FunctionProtoType::ExtParameterInfo, 16> paramInfos;
  RequiredArgs required = RequiredArgs::forPrototypePlus(ftp, prefix.size());
  // FIXME: Kill copy. -- from codegen
  appendParameterTypes(cgt, prefix, paramInfos, ftp);
  CanQualType resultType = ftp->getReturnType().getUnqualifiedType();

  return cgt.arrangeCIRFunctionInfo(resultType, instanceMethod, prefix,
                                    ftp->getExtInfo(), paramInfos, required);
}

/// Arrange the argument and result information for a value of the given
/// freestanding function type.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> ftp) {
  SmallVector<CanQualType, 16> argTypes;
  return ::arrangeCIRFunctionInfo(*this, FnInfoOpts::None, argTypes, ftp);
}

/// Arrange the argument and result information for a value of the given
/// unprototyped freestanding function type.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFreeFunctionType(CanQual<FunctionNoProtoType> ftnp) {
  // When translating an unprototyped function type, always use a
  // variadic type.
  return arrangeCIRFunctionInfo(ftnp->getReturnType().getUnqualifiedType(),
                                FnInfoOpts::None, std::nullopt,
                                ftnp->getExtInfo(), {}, RequiredArgs(0));
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeBuiltinFunctionCall(QualType resultType,
                                        const CallArgList &args) {
  // FIXME: Kill copy.
  SmallVector<CanQualType, 16> argTypes;
  for (const auto &arg : args)
    argTypes.push_back(getContext().getCanonicalParamType(arg.Ty));
  llvm_unreachable("NYI");
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// ExtraPrefixArgs is the number of ABI-specific args passed after the `this`
/// parameter.
/// ExtraSuffixArgs is the number of ABI-specific args passed at the end of
/// args.
/// PassProtoArgs indicates whether `args` has args for the parameters in the
/// given CXXConstructorDecl.
const CIRGenFunctionInfo &CIRGenTypes::arrangeCXXConstructorCall(
    const CallArgList &args, const CXXConstructorDecl *d, CXXCtorType ctorKind,
    unsigned extraPrefixArgs, unsigned extraSuffixArgs, bool passProtoArgs) {

  // FIXME: Kill copy.
  llvm::SmallVector<CanQualType, 16> argTypes;
  for (const auto &arg : args)
    argTypes.push_back(Context.getCanonicalParamType(arg.Ty));

  // +1 for implicit this, which should always be args[0]
  unsigned totalPrefixArgs = 1 + extraPrefixArgs;

  CanQual<FunctionProtoType> fpt = getFormalType(d);
  RequiredArgs required = passProtoArgs
                              ? RequiredArgs::forPrototypePlus(
                                    fpt, totalPrefixArgs + extraSuffixArgs)
                              : RequiredArgs::All;

  GlobalDecl gd(d, ctorKind);
  assert(!TheCXXABI.HasThisReturn(gd) && "ThisReturn NYI");
  assert(!TheCXXABI.hasMostDerivedReturn(gd) && "Most derived return NYI");
  CanQualType resultType = Context.VoidTy;

  FunctionType::ExtInfo info = fpt->getExtInfo();
  llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16> paramInfos;
  // If the prototype args are elided, we should onlyy have ABI-specific args,
  // which never have param info.
  assert(!fpt->hasExtParameterInfos() && "NYI");

  return arrangeCIRFunctionInfo(resultType, FnInfoOpts::IsInstanceMethod,
                                argTypes, info, paramInfos, required);
}

bool CIRGenTypes::inheritingCtorHasParams(const InheritedConstructor &inherited,
                                          CXXCtorType type) {

  // Parameters are unnecessary if we're constructing a base class subobject and
  // the inherited constructor lives in a virtual base.
  return type == Ctor_Complete ||
         !inherited.getShadowDecl()->constructsVirtualBase() ||
         !Target.getCXXABI().hasConstructorVariants();
}

bool CIRGenModule::MayDropFunctionReturn(const ASTContext &context,
                                         QualType returnType) {
  // We can't just disard the return value for a record type with a complex
  // destructor or a non-trivially copyable type.
  if (const RecordType *rt =
          returnType.getCanonicalType()->getAs<RecordType>()) {
    llvm_unreachable("NYI");
  }

  return returnType.isTriviallyCopyableType(context);
}

static bool isInAllocaArgument(CIRGenCXXABI &abi, QualType type) {
  const auto *rd = type->getAsCXXRecordDecl();
  return rd &&
         abi.getRecordArgABI(rd) == CIRGenCXXABI::RecordArgABI::DirectInMemory;
}

void CIRGenFunction::buildDelegateCallArg(CallArgList &args,
                                          const VarDecl *param,
                                          SourceLocation loc) {
  // StartFunction converted the ABI-lowered parameter(s) into a local alloca.
  // We need to turn that into an r-value suitable for buildCall
  Address local = GetAddrOfLocalVar(param);

  QualType type = param->getType();

  if (isInAllocaArgument(cgm.getCXXABI(), type)) {
    llvm_unreachable("NYI");
  }

  // GetAddrOfLocalVar returns a pointer-to-pointer for references, but the
  // argument needs to be the original pointer.
  if (type->isReferenceType()) {
    args.add(
        RValue::get(builder.createLoad(getLoc(param->getSourceRange()), local)),
        type);
  } else if (getLangOpts().ObjCAutoRefCount) {
    llvm_unreachable("NYI");
    // For the most part, we just need to load the alloca, except that aggregate
    // r-values are actually pointers to temporaries.
  } else {
    args.add(convertTempToRValue(local, type, loc), type);
  }

  // Deactivate the cleanup for the callee-destructed param that was pushed.
  if (type->isRecordType() && !CurFuncIsThunk &&
      type->castAs<RecordType>()->getDecl()->isParamDestroyedInCallee() &&
      param->needsDestruction(getContext())) {
    llvm_unreachable("NYI");
  }
}

/// Returns the "extra-canonicalized" return type, which discards qualifiers on
/// the return type. Codegen doesn't care about them, and it makes ABI code a
/// little easier to be able to assume that all parameter and return types are
/// top-level unqualified.
/// FIXME(CIR): This should be a common helper extracted from CodeGen
static CanQualType getReturnType(QualType retTy) {
  return retTy->getCanonicalTypeUnqualified().getUnqualifiedType();
}

/// Arrange a call as unto a free function, except possibly with an additional
/// number of formal parameters considered required.
static const CIRGenFunctionInfo &
arrangeFreeFunctionLikeCall(CIRGenTypes &cgt, CIRGenModule &cgm,
                            const CallArgList &args, const FunctionType *fnType,
                            unsigned numExtraRequiredArgs,
                            FnInfoOpts chainCall) {
  assert(args.size() >= numExtraRequiredArgs);
  assert((chainCall != FnInfoOpts::IsChainCall) && "Chain call NYI");

  llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16> paramInfos;

  // In most cases, there are no optional arguments.
  RequiredArgs required = RequiredArgs::All;

  // If we have a variadic prototype, the required arguments are the
  // extra prefix plus the arguments in the prototype.
  if (const FunctionProtoType *proto = dyn_cast<FunctionProtoType>(fnType)) {
    if (proto->isVariadic())
      required = RequiredArgs::forPrototypePlus(proto, numExtraRequiredArgs);

    if (proto->hasExtParameterInfos())
      addExtParameterInfosForCall(paramInfos, proto, numExtraRequiredArgs,
                                  args.size());
  } else if (llvm::isa<FunctionNoProtoType>(fnType)) {
    assert(!MissingFeatures::targetCodeGenInfoIsProtoCallVariadic());
    required = RequiredArgs(args.size());
  }

  // FIXME: Kill copy.
  SmallVector<CanQualType, 16> argTypes;
  for (const auto &arg : args)
    argTypes.push_back(cgt.getContext().getCanonicalParamType(arg.Ty));
  return cgt.arrangeCIRFunctionInfo(getReturnType(fnType->getReturnType()),
                                    chainCall, argTypes, fnType->getExtInfo(),
                                    paramInfos, required);
}

static llvm::SmallVector<CanQualType, 16>
getArgTypesForCall(ASTContext &ctx, const CallArgList &args) {
  llvm::SmallVector<CanQualType, 16> argTypes;
  for (auto &arg : args)
    argTypes.push_back(ctx.getCanonicalParamType(arg.Ty));
  return argTypes;
}

static llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16>
getExtParameterInfosForCall(const FunctionProtoType *proto, unsigned prefixArgs,
                            unsigned totalArgs) {
  llvm::SmallVector<FunctionProtoType::ExtParameterInfo, 16> result;
  if (proto->hasExtParameterInfos()) {
    llvm_unreachable("NYI");
  }
  return result;
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// numPrefixArgs is the number of the ABI-specific prefix arguments we have. It
/// does not count `this`.
const CIRGenFunctionInfo &CIRGenTypes::arrangeCXXMethodCall(
    const CallArgList &args, const FunctionProtoType *proto,
    RequiredArgs required, unsigned numPrefixArgs) {
  assert(numPrefixArgs + 1 <= args.size() &&
         "Emitting a call with less args than the required prefix?");
  // Add one to account for `this`. It is a bit awkard here, but we don't count
  // `this` in similar places elsewhere.
  auto paramInfos =
      getExtParameterInfosForCall(proto, numPrefixArgs + 1, args.size());

  // FIXME: Kill copy.
  auto argTypes = getArgTypesForCall(Context, args);

  auto info = proto->getExtInfo();
  return arrangeCIRFunctionInfo(getReturnType(proto->getReturnType()),
                                FnInfoOpts::IsInstanceMethod, argTypes, info,
                                paramInfos, required);
}

/// Figure out the rules for calling a function with the given formal type using
/// the given arguments. The arguments are necessary because the function might
/// be unprototyped, in which case it's target-dependent in crazy ways.
const CIRGenFunctionInfo &CIRGenTypes::arrangeFreeFunctionCall(
    const CallArgList &args, const FunctionType *fnType, bool chainCall) {
  assert(!chainCall && "ChainCall NYI");
  return arrangeFreeFunctionLikeCall(
      *this, CGM, args, fnType, chainCall ? 1 : 0,
      chainCall ? FnInfoOpts::IsChainCall : FnInfoOpts::None);
}

/// Set calling convention for CUDA/HIP kernel.
static void setCUDAKernelCallingConvention(CanQualType &fTy, CIRGenModule &cgm,
                                           const FunctionDecl *fd) {
  if (fd->hasAttr<CUDAGlobalAttr>()) {
    llvm_unreachable("NYI");
  }
}

/// Arrange the argument and result information for a declaration or definition
/// of the given C++ non-static member function. The member function must be an
/// ordinary function, i.e. not a constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodDeclaration(const CXXMethodDecl *md) {
  assert(!isa<CXXConstructorDecl>(md) && "wrong method for constructors!");
  assert(!isa<CXXDestructorDecl>(md) && "wrong method for destructors!");

  CanQualType ft = getFormalType(md).getAs<Type>();
  setCUDAKernelCallingConvention(ft, CGM, md);
  auto prototype = ft.getAs<FunctionProtoType>();

  if (md->isInstance()) {
    // The abstarct case is perfectly fine.
    auto *thisType = TheCXXABI.getThisArgumentTypeForMethod(md);
    return arrangeCXXMethodType(thisType, prototype.getTypePtr(), md);
  }

  return arrangeFreeFunctionType(prototype);
}

/// Arrange the argument and result information for a call to an unknown C++
/// non-static member function of the given abstract type. (A null RD means we
/// don't have any meaningful "this" argument type, so fall back to a generic
/// pointer type). The member fucntion must be an ordinary function, i.e. not a
/// constructor or destructor.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeCXXMethodType(const CXXRecordDecl *rd,
                                  const FunctionProtoType *ftp,
                                  const CXXMethodDecl *md) {
  llvm::SmallVector<CanQualType, 16> argTypes;

  // Add the 'this' pointer.
  argTypes.push_back(DeriveThisType(rd, md));

  return ::arrangeCIRFunctionInfo(
      *this, FnInfoOpts::IsChainCall, argTypes,
      ftp->getCanonicalTypeUnqualified().getAs<FunctionProtoType>());
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const CIRGenFunctionInfo &
CIRGenTypes::arrangeFunctionDeclaration(const FunctionDecl *fd) {
  if (const auto *md = dyn_cast<CXXMethodDecl>(fd))
    if (md->isInstance())
      return arrangeCXXMethodDeclaration(md);

  auto fTy = fd->getType()->getCanonicalTypeUnqualified();

  assert(isa<FunctionType>(fTy));
  // TODO: setCUDAKernelCallingConvention

  // When declaring a function without a prototype, always use a non-variadic
  // type.
  if (CanQual<FunctionNoProtoType> noProto = fTy.getAs<FunctionNoProtoType>()) {
    return arrangeCIRFunctionInfo(noProto->getReturnType(), FnInfoOpts::None,
                                  std::nullopt, noProto->getExtInfo(), {},
                                  RequiredArgs::All);
  }

  return arrangeFreeFunctionType(fTy.castAs<FunctionProtoType>());
}

RValue CallArg::getRValue(CIRGenFunction &cgf, mlir::Location loc) const {
  if (!HasLV)
    return RV;
  LValue copy = cgf.makeAddrLValue(cgf.CreateMemTemp(Ty, loc), Ty);
  cgf.buildAggregateCopy(copy, LV, Ty, AggValueSlot::DoesNotOverlap,
                         LV.isVolatile());
  IsUsed = true;
  return RValue::getAggregate(copy.getAddress());
}

void CIRGenFunction::buildNonNullArgCheck(RValue rv, QualType argType,
                                          SourceLocation argLoc,
                                          AbstractCallee ac, unsigned parmNum) {
  if (!ac.getDecl() || !(SanOpts.has(SanitizerKind::NonnullAttribute) ||
                         SanOpts.has(SanitizerKind::NullabilityArg)))
    return;
  llvm_unreachable("non-null arg check is NYI");
}

/* VarArg handling */

// FIXME(cir): This completely abstracts away the ABI with a generic CIR Op. We
// need to decide how to handle va_arg target-specific codegen.
mlir::Value CIRGenFunction::buildVAArg(VAArgExpr *ve, Address &vaListAddr) {
  assert(!ve->isMicrosoftABI() && "NYI");
  auto loc = cgm.getLoc(ve->getExprLoc());
  auto type = ConvertType(ve->getType());
  auto vaList = buildVAListRef(ve->getSubExpr()).getPointer();
  return builder.create<mlir::cir::VAArgOp>(loc, type, vaList);
}

static void getTrivialDefaultFunctionAttributes(
    StringRef name, bool hasOptnone, const CodeGenOptions &codeGenOpts,
    const LangOptions &langOpts, bool attrOnCallSite, CIRGenModule &cgm,
    mlir::NamedAttrList &funcAttrs) {

  if (langOpts.assumeFunctionsAreConvergent()) {
    // Conservatively, mark all functions and calls in CUDA and OpenCL as
    // convergent (meaning, they may call an intrinsically convergent op, such
    // as __syncthreads() / barrier(), and so can't have certain optimizations
    // applied around them).  LLVM will remove this attribute where it safely
    // can.

    auto convgt = mlir::cir::ConvergentAttr::get(cgm.getBuilder().getContext());
    funcAttrs.set(convgt.getMnemonic(), convgt);
  }

  // TODO: NoThrow attribute should be added for other GPU modes CUDA, SYCL,
  // HIP, OpenMP offload.
  // AFAIK, neither of them support exceptions in device code.
  if ((langOpts.CUDA && langOpts.CUDAIsDevice) || langOpts.SYCLIsDevice)
    llvm_unreachable("NYI");
  if (langOpts.OpenCL) {
    auto noThrow = mlir::cir::NoThrowAttr::get(cgm.getBuilder().getContext());
    funcAttrs.set(noThrow.getMnemonic(), noThrow);
  }
}

void CIRGenModule::getTrivialDefaultFunctionAttributes(
    StringRef name, bool hasOptnone, bool attrOnCallSite,
    mlir::NamedAttrList &funcAttrs) {
  ::getTrivialDefaultFunctionAttributes(name, hasOptnone, getCodeGenOpts(),
                                        getLangOpts(), attrOnCallSite, *this,
                                        funcAttrs);
}

void CIRGenModule::getDefaultFunctionAttributes(
    StringRef name, bool hasOptnone, bool attrOnCallSite,
    mlir::NamedAttrList &funcAttrs) {
  getTrivialDefaultFunctionAttributes(name, hasOptnone, attrOnCallSite,
                                      funcAttrs);
  // If we're just getting the default, get the default values for mergeable
  // attributes.
  if (!attrOnCallSite) {
    // TODO(cir): addMergableDefaultFunctionAttributes(codeGenOpts, funcAttrs);
  }
}
