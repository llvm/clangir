//===- LoweringPrepare.cpp - pareparation work for LLVM lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Address.h"
#include "CIRGenBuilder.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/LoweringPrepareCXXABI.h"
#include "clang/CIR/Dialect/Transforms/PassDetail.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

#include <memory>

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace cir;

static SmallString<128> getTransformedFileName(ModuleOp theModule) {
  SmallString<128> FileName;

  if (theModule.getSymName()) {
    FileName = llvm::sys::path::filename(theModule.getSymName()->str());
  }

  if (FileName.empty())
    FileName = "<null>";

  for (size_t i = 0; i < FileName.size(); ++i) {
    // Replace everything that's not [a-zA-Z0-9._] with a _. This set happens
    // to be the set of C preprocessing numbers.
    if (!clang::isPreprocessingNumberBody(FileName[i]))
      FileName[i] = '_';
  }

  return FileName;
}

/// Return the FuncOp called by `callOp`.
static FuncOp getCalledFunction(CallOp callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

namespace {

struct LoweringPreparePass : public LoweringPrepareBase<LoweringPreparePass> {
  LoweringPreparePass() = default;
  void runOnOperation() override;

  void runOnOp(Operation *op);
  void lowerUnaryOp(UnaryOp op);
  void lowerBinOp(BinOp op);
  void lowerCastOp(CastOp op);
  void lowerComplexBinOp(ComplexBinOp op);
  void lowerThreeWayCmpOp(CmpThreeWayOp op);
  void lowerVAArgOp(VAArgOp op);
  void lowerGlobalOp(GlobalOp op);
  void lowerGetGlobalOp(GetGlobalOp op);
  void lowerDynamicCastOp(DynamicCastOp op);
  void lowerStdFindOp(StdFindOp op);
  void lowerIterBeginOp(IterBeginOp op);
  void lowerIterEndOp(IterEndOp op);
  void lowerToMemCpy(StoreOp op);
  void lowerArrayDtor(ArrayDtor op);
  void lowerArrayCtor(ArrayCtor op);

  void handleStaticLocal(GlobalOp globalOp, GetGlobalOp getGlobalOp);
  void handleGlobalOpCtorDtor(GlobalOp globalOp);

  /// Collect annotations of global values in the module
  void addGlobalAnnotations(mlir::Operation *op, mlir::ArrayAttr annotations);

  /// Build the function that initializes the specified global
  FuncOp buildCXXGlobalVarDeclInitFunc(GlobalOp op);

  /// Build a module init function that calls all the dynamic initializers.
  void buildCXXGlobalInitFunc();

  /// Materialize global ctor/dtor list
  void buildGlobalCtorDtorList();

  /// Build attribute of global annotation values
  void buildGlobalAnnotationValues();

  cir::GlobalOp
  getStaticLocalDeclGuardAddress(cir::ASTVarDeclInterface varDecl) {
    return staticLocalDeclGuardMap[varDecl];
  }

  void setStaticLocalDeclGuardAddress(cir::ASTVarDeclInterface varDecl,
                                      cir::GlobalOp globalOp) {
    staticLocalDeclGuardMap[varDecl] = globalOp;
  }

  FuncOp buildRuntimeFunction(
      mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
      cir::FuncType type,
      cir::GlobalLinkageKind linkage = cir::GlobalLinkageKind::ExternalLinkage);

  GlobalOp buildRuntimeVariable(
      mlir::OpBuilder &Builder, llvm::StringRef Name, mlir::Location Loc,
      mlir::Type type,
      cir::GlobalLinkageKind Linkage = cir::GlobalLinkageKind::ExternalLinkage);

  /// Track the current number of global array string count for when the symbol
  /// has an empty name, and prevent collisions.
  uint64_t annonGlobalConstArrayCount = 0;

  ///
  /// AST related
  /// -----------

  clang::ASTContext *astCtx;
  std::shared_ptr<cir::LoweringPrepareCXXABI> cxxABI;
  clang::CIRGen::CIRGenBuilderTy *builder;

  void setASTContext(clang::ASTContext *c) {
    astCtx = c;
    const clang::TargetInfo &target = c->getTargetInfo();
    auto abiStr = target.getABI();
    switch (c->getCXXABIKind()) {
    case clang::TargetCXXABI::GenericItanium:
      if (target.getTriple().getArch() == llvm::Triple::x86_64) {
        cxxABI.reset(
            cir::LoweringPrepareCXXABI::createX86ABI(/*is64bit=*/true));
        break;
      }

      cxxABI.reset(cir::LoweringPrepareCXXABI::createItaniumABI());
      break;
    case clang::TargetCXXABI::GenericAArch64:
    case clang::TargetCXXABI::AppleARM64:
      // TODO: This is temporary solution. ABIKind info should be
      // propagated from the targetInfo managed by ABI lowering
      // query system.
      assert(abiStr == "aapcs" || abiStr == "darwinpcs" ||
             abiStr == "aapcs-soft");
      cxxABI.reset(cir::LoweringPrepareCXXABI::createAArch64ABI(
          abiStr == "aapcs"
              ? cir::AArch64ABIKind::AAPCS
              : (abiStr == "darwinpccs" ? cir::AArch64ABIKind::DarwinPCS
                                        : cir::AArch64ABIKind::AAPCSSoft)));
      break;
    default:
      llvm_unreachable("NYI");
    }
  }

  void setBuilder(clang::CIRGen::CIRGenBuilderTy &builder) {
    this->builder = &builder;
  }

  /// Tracks current module.
  ModuleOp theModule;

  /// Tracks existing dynamic initializers.
  llvm::StringMap<uint32_t> dynamicInitializerNames;
  llvm::SmallVector<FuncOp, 4> dynamicInitializers;

  /// List of ctors to be called before main()
  llvm::SmallVector<mlir::Attribute, 4> globalCtorList;
  /// List of dtors to be called when unloading module.
  llvm::SmallVector<mlir::Attribute, 4> globalDtorList;
  /// List of annotations in the module
  llvm::SmallVector<mlir::Attribute, 4> globalAnnotations;

  llvm::DenseMap<cir::ASTVarDeclInterface, cir::GlobalOp>
      staticLocalDeclGuardMap;
};
} // namespace

GlobalOp LoweringPreparePass::buildRuntimeVariable(
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
    mlir::Type type, cir::GlobalLinkageKind linkage) {
  GlobalOp g = dyn_cast_or_null<GlobalOp>(SymbolTable::lookupNearestSymbolFrom(
      theModule, StringAttr::get(theModule->getContext(), name)));
  if (!g) {
    g = builder.create<cir::GlobalOp>(loc, name, type);
    g.setLinkageAttr(
        cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  return g;
}

FuncOp LoweringPreparePass::buildRuntimeFunction(
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
    cir::FuncType type, cir::GlobalLinkageKind linkage) {
  FuncOp f = dyn_cast_or_null<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      theModule, StringAttr::get(theModule->getContext(), name)));
  if (!f) {
    f = builder.create<cir::FuncOp>(loc, name, type);
    f.setLinkageAttr(
        cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);
    mlir::NamedAttrList attrs;
    f.setExtraAttrsAttr(cir::ExtraFuncAttributesAttr::get(
        builder.getContext(), attrs.getDictionary(builder.getContext())));
  }
  return f;
}

FuncOp LoweringPreparePass::buildCXXGlobalVarDeclInitFunc(GlobalOp op) {
  SmallString<256> fnName;
  {
    llvm::raw_svector_ostream Out(fnName);
    op.getAst()->mangleDynamicInitializer(Out);
    // Name numbering
    uint32_t cnt = dynamicInitializerNames[fnName]++;
    if (cnt)
      fnName += "." + llvm::Twine(cnt).str();
  }

  // Create a variable initialization function.
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);
  auto voidTy = cir::VoidType::get(builder.getContext());
  auto fnType = cir::FuncType::get({}, voidTy);
  FuncOp f = buildRuntimeFunction(builder, fnName, op.getLoc(), fnType,
                                  cir::GlobalLinkageKind::InternalLinkage);

  // Move over the initialzation code of the ctor region.
  auto &block = op.getCtorRegion().front();
  mlir::Block *entryBB = f.addEntryBlock();
  entryBB->getOperations().splice(entryBB->begin(), block.getOperations(),
                                  block.begin(), std::prev(block.end()));

  // Register the destructor call with __cxa_atexit
  auto &dtorRegion = op.getDtorRegion();
  if (!dtorRegion.empty()) {
    assert(op.getAst() &&
           op.getAst()->getTLSKind() == clang::VarDecl::TLS_None && " TLS NYI");
    // Create a variable that binds the atexit to this shared object.
    builder.setInsertionPointToStart(&theModule.getBodyRegion().front());
    auto Handle = buildRuntimeVariable(builder, "__dso_handle", op.getLoc(),
                                       builder.getI8Type());

    // Look for the destructor call in dtorBlock
    auto &dtorBlock = dtorRegion.front();
    cir::CallOp dtorCall;
    for (auto op : reverse(dtorBlock.getOps<cir::CallOp>())) {
      dtorCall = op;
      break;
    }
    assert(dtorCall && "Expected a dtor call");
    FuncOp dtorFunc = getCalledFunction(dtorCall);
    assert(dtorFunc &&
           mlir::isa<ASTCXXDestructorDeclInterface>(*dtorFunc.getAst()) &&
           "Expected a dtor call");

    // Create a runtime helper function:
    //    extern "C" int __cxa_atexit(void (*f)(void *), void *p, void *d);
    auto voidPtrTy = cir::PointerType::get(builder.getContext(), voidTy);
    auto voidFnTy = cir::FuncType::get({voidPtrTy}, voidTy);
    auto voidFnPtrTy = cir::PointerType::get(builder.getContext(), voidFnTy);
    auto HandlePtrTy =
        cir::PointerType::get(builder.getContext(), Handle.getSymType());
    auto fnAtExitType =
        cir::FuncType::get({voidFnPtrTy, voidPtrTy, HandlePtrTy},
                           cir::VoidType::get(builder.getContext()));
    const char *nameAtExit = "__cxa_atexit";
    FuncOp fnAtExit =
        buildRuntimeFunction(builder, nameAtExit, op.getLoc(), fnAtExitType);

    // Replace the dtor call with a call to __cxa_atexit(&dtor, &var,
    // &__dso_handle)
    builder.setInsertionPointAfter(dtorCall);
    mlir::Value args[3];
    auto dtorPtrTy =
        cir::PointerType::get(builder.getContext(), dtorFunc.getFunctionType());
    // dtorPtrTy
    args[0] = builder.create<cir::GetGlobalOp>(dtorCall.getLoc(), dtorPtrTy,
                                               dtorFunc.getSymName());
    args[0] = builder.create<cir::CastOp>(dtorCall.getLoc(), voidFnPtrTy,
                                          cir::CastKind::bitcast, args[0]);
    args[1] = builder.create<cir::CastOp>(dtorCall.getLoc(), voidPtrTy,
                                          cir::CastKind::bitcast,
                                          dtorCall.getArgOperand(0));
    args[2] = builder.create<cir::GetGlobalOp>(Handle.getLoc(), HandlePtrTy,
                                               Handle.getSymName());
    builder.createCallOp(dtorCall.getLoc(), fnAtExit, args);
    dtorCall->erase();
    entryBB->getOperations().splice(entryBB->end(), dtorBlock.getOperations(),
                                    dtorBlock.begin(),
                                    std::prev(dtorBlock.end()));
  }

  // Replace cir.yield with cir.return
  builder.setInsertionPointToEnd(entryBB);
  auto &yieldOp = block.getOperations().back();
  assert(isa<YieldOp>(yieldOp));
  builder.create<ReturnOp>(yieldOp.getLoc());
  return f;
}

static void canonicalizeIntrinsicThreeWayCmp(CIRBaseBuilderTy &builder,
                                             CmpThreeWayOp op) {
  auto loc = op->getLoc();
  auto cmpInfo = op.getInfo();

  if (cmpInfo.getLt() == -1 && cmpInfo.getEq() == 0 && cmpInfo.getGt() == 1) {
    // The comparison is already in canonicalized form.
    return;
  }

  auto canonicalizedCmpInfo =
      cir::CmpThreeWayInfoAttr::get(builder.getContext(), -1, 0, 1);
  mlir::Value result =
      builder
          .create<cir::CmpThreeWayOp>(loc, op.getType(), op.getLhs(),
                                      op.getRhs(), canonicalizedCmpInfo)
          .getResult();

  auto compareAndYield = [&](mlir::Value input, int64_t test,
                             int64_t yield) -> mlir::Value {
    // Create a conditional branch that tests whether `input` is equal to
    // `test`. If `input` is equal to `test`, yield `yield`. Otherwise, yield
    // `input` as is.
    auto testValue =
        builder.getConstant(loc, cir::IntAttr::get(input.getType(), test));
    auto yieldValue =
        builder.getConstant(loc, cir::IntAttr::get(input.getType(), yield));
    auto eqToTest =
        builder.createCompare(loc, cir::CmpOpKind::eq, input, testValue);
    return builder.createSelect(loc, eqToTest, yieldValue, input);
  };

  if (cmpInfo.getLt() != -1)
    result = compareAndYield(result, -1, cmpInfo.getLt());

  if (cmpInfo.getEq() != 0)
    result = compareAndYield(result, 0, cmpInfo.getEq());

  if (cmpInfo.getGt() != 1)
    result = compareAndYield(result, 1, cmpInfo.getGt());

  op.replaceAllUsesWith(result);
  op.erase();
}

void LoweringPreparePass::lowerVAArgOp(VAArgOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPoint(op);
  cir::CIRDataLayout datalayout(theModule);

  auto res = cxxABI->lowerVAArg(builder, op, datalayout);
  if (res) {
    op.replaceAllUsesWith(res);
    op.erase();
  }
}

void LoweringPreparePass::lowerUnaryOp(UnaryOp op) {
  auto ty = op.getType();
  if (!mlir::isa<cir::ComplexType>(ty))
    return;

  auto loc = op.getLoc();
  auto opKind = op.getKind();

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  auto operand = op.getInput();

  auto operandReal = builder.createComplexReal(loc, operand);
  auto operandImag = builder.createComplexImag(loc, operand);

  mlir::Value resultReal;
  mlir::Value resultImag;
  switch (opKind) {
  case cir::UnaryOpKind::Inc:
  case cir::UnaryOpKind::Dec:
    resultReal = builder.createUnaryOp(loc, opKind, operandReal);
    resultImag = operandImag;
    break;

  case cir::UnaryOpKind::Plus:
  case cir::UnaryOpKind::Minus:
    resultReal = builder.createUnaryOp(loc, opKind, operandReal);
    resultImag = builder.createUnaryOp(loc, opKind, operandImag);
    break;

  case cir::UnaryOpKind::Not:
    resultReal = operandReal;
    resultImag =
        builder.createUnaryOp(loc, cir::UnaryOpKind::Minus, operandImag);
    break;
  }

  auto result = builder.createComplexCreate(loc, resultReal, resultImag);
  op.replaceAllUsesWith(result);
  op.erase();
}

void LoweringPreparePass::lowerBinOp(BinOp op) {
  auto ty = op.getType();
  if (!mlir::isa<cir::ComplexType>(ty))
    return;

  auto loc = op.getLoc();
  auto opKind = op.getKind();
  assert((opKind == cir::BinOpKind::Add || opKind == cir::BinOpKind::Sub) &&
         "invalid binary op kind on complex numbers");

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();

  // (a+bi) + (c+di) = (a+c) + (b+d)i
  // (a+bi) - (c+di) = (a-c) + (b-d)i
  auto lhsReal = builder.createComplexReal(loc, lhs);
  auto lhsImag = builder.createComplexImag(loc, lhs);
  auto rhsReal = builder.createComplexReal(loc, rhs);
  auto rhsImag = builder.createComplexImag(loc, rhs);
  auto resultReal = builder.createBinop(lhsReal, opKind, rhsReal);
  auto resultImag = builder.createBinop(lhsImag, opKind, rhsImag);
  auto result = builder.createComplexCreate(loc, resultReal, resultImag);

  op.replaceAllUsesWith(result);
  op.erase();
}

static mlir::Value lowerScalarToComplexCast(MLIRContext &ctx, CastOp op) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  auto src = op.getSrc();
  auto imag = builder.getNullValue(src.getType(), op.getLoc());
  return builder.createComplexCreate(op.getLoc(), src, imag);
}

static mlir::Value lowerComplexToScalarCast(MLIRContext &ctx, CastOp op) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  auto src = op.getSrc();

  if (!mlir::isa<cir::BoolType>(op.getType()))
    return builder.createComplexReal(op.getLoc(), src);

  // Complex cast to bool: (bool)(a+bi) => (bool)a || (bool)b
  auto srcReal = builder.createComplexReal(op.getLoc(), src);
  auto srcImag = builder.createComplexImag(op.getLoc(), src);

  cir::CastKind elemToBoolKind;
  if (op.getKind() == cir::CastKind::float_complex_to_bool)
    elemToBoolKind = cir::CastKind::float_to_bool;
  else if (op.getKind() == cir::CastKind::int_complex_to_bool)
    elemToBoolKind = cir::CastKind::int_to_bool;
  else
    llvm_unreachable("invalid complex to bool cast kind");

  auto boolTy = builder.getBoolTy();
  auto srcRealToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcReal, boolTy);
  auto srcImagToBool =
      builder.createCast(op.getLoc(), elemToBoolKind, srcImag, boolTy);

  // srcRealToBool || srcImagToBool
  return builder.createLogicalOr(op.getLoc(), srcRealToBool, srcImagToBool);
}

static mlir::Value lowerComplexToComplexCast(MLIRContext &ctx, CastOp op) {
  CIRBaseBuilderTy builder(ctx);
  builder.setInsertionPoint(op);

  auto src = op.getSrc();
  auto dstComplexElemTy =
      mlir::cast<cir::ComplexType>(op.getType()).getElementTy();

  auto srcReal = builder.createComplexReal(op.getLoc(), src);
  auto srcImag = builder.createComplexReal(op.getLoc(), src);

  cir::CastKind scalarCastKind;
  switch (op.getKind()) {
  case cir::CastKind::float_complex:
    scalarCastKind = cir::CastKind::floating;
    break;
  case cir::CastKind::float_complex_to_int_complex:
    scalarCastKind = cir::CastKind::float_to_int;
    break;
  case cir::CastKind::int_complex:
    scalarCastKind = cir::CastKind::integral;
    break;
  case cir::CastKind::int_complex_to_float_complex:
    scalarCastKind = cir::CastKind::int_to_float;
    break;
  default:
    llvm_unreachable("invalid complex to complex cast kind");
  }

  auto dstReal = builder.createCast(op.getLoc(), scalarCastKind, srcReal,
                                    dstComplexElemTy);
  auto dstImag = builder.createCast(op.getLoc(), scalarCastKind, srcImag,
                                    dstComplexElemTy);
  return builder.createComplexCreate(op.getLoc(), dstReal, dstImag);
}

void LoweringPreparePass::lowerCastOp(CastOp op) {
  mlir::Value loweredValue;
  switch (op.getKind()) {
  case cir::CastKind::float_to_complex:
  case cir::CastKind::int_to_complex:
    loweredValue = lowerScalarToComplexCast(getContext(), op);
    break;

  case cir::CastKind::float_complex_to_real:
  case cir::CastKind::int_complex_to_real:
  case cir::CastKind::float_complex_to_bool:
  case cir::CastKind::int_complex_to_bool:
    loweredValue = lowerComplexToScalarCast(getContext(), op);
    break;

  case cir::CastKind::float_complex:
  case cir::CastKind::float_complex_to_int_complex:
  case cir::CastKind::int_complex:
  case cir::CastKind::int_complex_to_float_complex:
    loweredValue = lowerComplexToComplexCast(getContext(), op);
    break;

  default:
    return;
  }

  op.replaceAllUsesWith(loweredValue);
  op.erase();
}

static mlir::Value buildComplexBinOpLibCall(
    LoweringPreparePass &pass, CIRBaseBuilderTy &builder,
    llvm::StringRef (*libFuncNameGetter)(llvm::APFloat::Semantics),
    mlir::Location loc, cir::ComplexType ty, mlir::Value lhsReal,
    mlir::Value lhsImag, mlir::Value rhsReal, mlir::Value rhsImag) {
  auto elementTy = mlir::cast<cir::CIRFPTypeInterface>(ty.getElementTy());

  auto libFuncName = libFuncNameGetter(
      llvm::APFloat::SemanticsToEnum(elementTy.getFloatSemantics()));
  llvm::SmallVector<mlir::Type, 4> libFuncInputTypes(4, elementTy);
  auto libFuncTy = cir::FuncType::get(libFuncInputTypes, ty);

  cir::FuncOp libFunc;
  {
    mlir::OpBuilder::InsertionGuard ipGuard{builder};
    builder.setInsertionPointToStart(pass.theModule.getBody());
    libFunc = pass.buildRuntimeFunction(builder, libFuncName, loc, libFuncTy);
  }

  auto call =
      builder.createCallOp(loc, libFunc, {lhsReal, lhsImag, rhsReal, rhsImag});
  return call.getResult();
}

static llvm::StringRef
getComplexMulLibCallName(llvm::APFloat::Semantics semantics) {
  switch (semantics) {
  case llvm::APFloat::S_IEEEhalf:
    return "__mulhc3";
  case llvm::APFloat::S_IEEEsingle:
    return "__mulsc3";
  case llvm::APFloat::S_IEEEdouble:
    return "__muldc3";
  case llvm::APFloat::S_PPCDoubleDouble:
    return "__multc3";
  case llvm::APFloat::S_x87DoubleExtended:
    return "__mulxc3";
  case llvm::APFloat::S_IEEEquad:
    return "__multc3";
  default:
    llvm_unreachable("unsupported floating point type");
  }
}

static llvm::StringRef
getComplexDivLibCallName(llvm::APFloat::Semantics semantics) {
  switch (semantics) {
  case llvm::APFloat::S_IEEEhalf:
    return "__divhc3";
  case llvm::APFloat::S_IEEEsingle:
    return "__divsc3";
  case llvm::APFloat::S_IEEEdouble:
    return "__divdc3";
  case llvm::APFloat::S_PPCDoubleDouble:
    return "__divtc3";
  case llvm::APFloat::S_x87DoubleExtended:
    return "__divxc3";
  case llvm::APFloat::S_IEEEquad:
    return "__divtc3";
  default:
    llvm_unreachable("unsupported floating point type");
  }
}

static mlir::Value lowerComplexMul(LoweringPreparePass &pass,
                                   CIRBaseBuilderTy &builder,
                                   mlir::Location loc, cir::ComplexBinOp op,
                                   mlir::Value lhsReal, mlir::Value lhsImag,
                                   mlir::Value rhsReal, mlir::Value rhsImag) {
  // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
  auto resultRealLhs =
      builder.createBinop(lhsReal, cir::BinOpKind::Mul, rhsReal);
  auto resultRealRhs =
      builder.createBinop(lhsImag, cir::BinOpKind::Mul, rhsImag);
  auto resultImagLhs =
      builder.createBinop(lhsReal, cir::BinOpKind::Mul, rhsImag);
  auto resultImagRhs =
      builder.createBinop(lhsImag, cir::BinOpKind::Mul, rhsReal);
  auto resultReal =
      builder.createBinop(resultRealLhs, cir::BinOpKind::Sub, resultRealRhs);
  auto resultImag =
      builder.createBinop(resultImagLhs, cir::BinOpKind::Add, resultImagRhs);
  auto algebraicResult =
      builder.createComplexCreate(loc, resultReal, resultImag);

  auto ty = op.getType();
  auto range = op.getRange();
  if (mlir::isa<cir::IntType>(ty.getElementTy()) ||
      range == cir::ComplexRangeKind::Basic ||
      range == cir::ComplexRangeKind::Improved ||
      range == cir::ComplexRangeKind::Promoted)
    return algebraicResult;

  // Check whether the real part and the imaginary part of the result are both
  // NaN. If so, emit a library call to compute the multiplication instead.
  // We check a value against NaN by comparing the value against itself.
  auto resultRealIsNaN = builder.createIsNaN(loc, resultReal);
  auto resultImagIsNaN = builder.createIsNaN(loc, resultImag);
  auto resultRealAndImagAreNaN =
      builder.createLogicalAnd(loc, resultRealIsNaN, resultImagIsNaN);
  return builder
      .create<cir::TernaryOp>(
          loc, resultRealAndImagAreNaN,
          [&](mlir::OpBuilder &, mlir::Location) {
            auto libCallResult = buildComplexBinOpLibCall(
                pass, builder, &getComplexMulLibCallName, loc, ty, lhsReal,
                lhsImag, rhsReal, rhsImag);
            builder.createYield(loc, libCallResult);
          },
          [&](mlir::OpBuilder &, mlir::Location) {
            builder.createYield(loc, algebraicResult);
          })
      .getResult();
}

static mlir::Value
buildAlgebraicComplexDiv(CIRBaseBuilderTy &builder, mlir::Location loc,
                         mlir::Value lhsReal, mlir::Value lhsImag,
                         mlir::Value rhsReal, mlir::Value rhsImag) {
  // (a+bi) / (c+di) = ((ac+bd)/(cc+dd)) + ((bc-ad)/(cc+dd))i
  auto &a = lhsReal;
  auto &b = lhsImag;
  auto &c = rhsReal;
  auto &d = rhsImag;

  auto ac = builder.createBinop(loc, a, cir::BinOpKind::Mul, c);     // a*c
  auto bd = builder.createBinop(loc, b, cir::BinOpKind::Mul, d);     // b*d
  auto cc = builder.createBinop(loc, c, cir::BinOpKind::Mul, c);     // c*c
  auto dd = builder.createBinop(loc, d, cir::BinOpKind::Mul, d);     // d*d
  auto acbd = builder.createBinop(loc, ac, cir::BinOpKind::Add, bd); // ac+bd
  auto ccdd = builder.createBinop(loc, cc, cir::BinOpKind::Add, dd); // cc+dd
  auto resultReal = builder.createBinop(loc, acbd, cir::BinOpKind::Div, ccdd);

  auto bc = builder.createBinop(loc, b, cir::BinOpKind::Mul, c);     // b*c
  auto ad = builder.createBinop(loc, a, cir::BinOpKind::Mul, d);     // a*d
  auto bcad = builder.createBinop(loc, bc, cir::BinOpKind::Sub, ad); // bc-ad
  auto resultImag = builder.createBinop(loc, bcad, cir::BinOpKind::Div, ccdd);

  return builder.createComplexCreate(loc, resultReal, resultImag);
}

static mlir::Value
buildRangeReductionComplexDiv(CIRBaseBuilderTy &builder, mlir::Location loc,
                              mlir::Value lhsReal, mlir::Value lhsImag,
                              mlir::Value rhsReal, mlir::Value rhsImag) {
  // Implements Smith's algorithm for complex division.
  // SMITH, R. L. Algorithm 116: Complex division. Commun. ACM 5, 8 (1962).

  // Let:
  //   - lhs := a+bi
  //   - rhs := c+di
  //   - result := lhs / rhs = e+fi
  //
  // The algorithm psudocode looks like follows:
  //   if fabs(c) >= fabs(d):
  //     r := d / c
  //     tmp := c + r*d
  //     e = (a + b*r) / tmp
  //     f = (b - a*r) / tmp
  //   else:
  //     r := c / d
  //     tmp := d + r*c
  //     e = (a*r + b) / tmp
  //     f = (b*r - a) / tmp

  auto &a = lhsReal;
  auto &b = lhsImag;
  auto &c = rhsReal;
  auto &d = rhsImag;

  auto trueBranchBuilder = [&](mlir::OpBuilder &, mlir::Location) {
    auto r = builder.createBinop(loc, d, cir::BinOpKind::Div,
                                 c);                               // r := d / c
    auto rd = builder.createBinop(loc, r, cir::BinOpKind::Mul, d); // r*d
    auto tmp = builder.createBinop(loc, c, cir::BinOpKind::Add,
                                   rd); // tmp := c + r*d

    auto br = builder.createBinop(loc, b, cir::BinOpKind::Mul, r);   // b*r
    auto abr = builder.createBinop(loc, a, cir::BinOpKind::Add, br); // a + b*r
    auto e = builder.createBinop(loc, abr, cir::BinOpKind::Div, tmp);

    auto ar = builder.createBinop(loc, a, cir::BinOpKind::Mul, r);   // a*r
    auto bar = builder.createBinop(loc, b, cir::BinOpKind::Sub, ar); // b - a*r
    auto f = builder.createBinop(loc, bar, cir::BinOpKind::Div, tmp);

    auto result = builder.createComplexCreate(loc, e, f);
    builder.createYield(loc, result);
  };

  auto falseBranchBuilder = [&](mlir::OpBuilder &, mlir::Location) {
    auto r = builder.createBinop(loc, c, cir::BinOpKind::Div,
                                 d);                               // r := c / d
    auto rc = builder.createBinop(loc, r, cir::BinOpKind::Mul, c); // r*c
    auto tmp = builder.createBinop(loc, d, cir::BinOpKind::Add,
                                   rc); // tmp := d + r*c

    auto ar = builder.createBinop(loc, a, cir::BinOpKind::Mul, r);   // a*r
    auto arb = builder.createBinop(loc, ar, cir::BinOpKind::Add, b); // a*r + b
    auto e = builder.createBinop(loc, arb, cir::BinOpKind::Div, tmp);

    auto br = builder.createBinop(loc, b, cir::BinOpKind::Mul, r);   // b*r
    auto bra = builder.createBinop(loc, br, cir::BinOpKind::Sub, a); // b*r - a
    auto f = builder.createBinop(loc, bra, cir::BinOpKind::Div, tmp);

    auto result = builder.createComplexCreate(loc, e, f);
    builder.createYield(loc, result);
  };

  auto cFabs = builder.create<cir::FAbsOp>(loc, c);
  auto dFabs = builder.create<cir::FAbsOp>(loc, d);
  auto cmpResult = builder.createCompare(loc, cir::CmpOpKind::ge, cFabs, dFabs);
  auto ternary = builder.create<cir::TernaryOp>(
      loc, cmpResult, trueBranchBuilder, falseBranchBuilder);

  return ternary.getResult();
}

static mlir::Value lowerComplexDiv(LoweringPreparePass &pass,
                                   CIRBaseBuilderTy &builder,
                                   mlir::Location loc, cir::ComplexBinOp op,
                                   mlir::Value lhsReal, mlir::Value lhsImag,
                                   mlir::Value rhsReal, mlir::Value rhsImag) {
  auto ty = op.getType();
  if (mlir::isa<cir::CIRFPTypeInterface>(ty.getElementTy())) {
    auto range = op.getRange();
    if (range == cir::ComplexRangeKind::Improved ||
        (range == cir::ComplexRangeKind::Promoted && !op.getPromoted()))
      return buildRangeReductionComplexDiv(builder, loc, lhsReal, lhsImag,
                                           rhsReal, rhsImag);
    if (range == cir::ComplexRangeKind::Full)
      return buildComplexBinOpLibCall(pass, builder, &getComplexDivLibCallName,
                                      loc, ty, lhsReal, lhsImag, rhsReal,
                                      rhsImag);
  }

  return buildAlgebraicComplexDiv(builder, loc, lhsReal, lhsImag, rhsReal,
                                  rhsImag);
}

void LoweringPreparePass::lowerComplexBinOp(ComplexBinOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  auto loc = op.getLoc();
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto lhsReal = builder.createComplexReal(loc, lhs);
  auto lhsImag = builder.createComplexImag(loc, lhs);
  auto rhsReal = builder.createComplexReal(loc, rhs);
  auto rhsImag = builder.createComplexImag(loc, rhs);

  mlir::Value loweredResult;
  if (op.getKind() == cir::ComplexBinOpKind::Mul)
    loweredResult = lowerComplexMul(*this, builder, loc, op, lhsReal, lhsImag,
                                    rhsReal, rhsImag);
  else
    loweredResult = lowerComplexDiv(*this, builder, loc, op, lhsReal, lhsImag,
                                    rhsReal, rhsImag);

  op.replaceAllUsesWith(loweredResult);
  op.erase();
}

void LoweringPreparePass::lowerThreeWayCmpOp(CmpThreeWayOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  if (op.isIntegralComparison() && op.isStrongOrdering()) {
    // For three-way comparisons on integral operands that produce strong
    // ordering, we can generate potentially better code with the `llvm.scmp.*`
    // and `llvm.ucmp.*` intrinsics. Thus we don't replace these comparisons
    // here. They will be lowered directly to LLVMIR during the LLVM lowering
    // pass.
    //
    // But we still need to take a step here. `llvm.scmp.*` and `llvm.ucmp.*`
    // returns -1, 0, or 1 to represent lt, eq, and gt, which are the
    // "canonicalized" result values of three-way comparisons. However,
    // `cir.cmp3way` may not produce canonicalized result. We need to
    // canonicalize the comparison if necessary. This is what we're doing in
    // this special branch.
    canonicalizeIntrinsicThreeWayCmp(builder, op);
    return;
  }

  auto loc = op->getLoc();
  auto cmpInfo = op.getInfo();

  auto buildCmpRes = [&](int64_t value) -> mlir::Value {
    return builder.create<cir::ConstantOp>(
        loc, op.getType(), cir::IntAttr::get(op.getType(), value));
  };
  auto ltRes = buildCmpRes(cmpInfo.getLt());
  auto eqRes = buildCmpRes(cmpInfo.getEq());
  auto gtRes = buildCmpRes(cmpInfo.getGt());

  auto buildCmp = [&](CmpOpKind kind) -> mlir::Value {
    auto ty = BoolType::get(&getContext());
    return builder.create<cir::CmpOp>(loc, ty, kind, op.getLhs(), op.getRhs());
  };
  auto buildSelect = [&](mlir::Value condition, mlir::Value trueResult,
                         mlir::Value falseResult) -> mlir::Value {
    return builder.createSelect(loc, condition, trueResult, falseResult);
  };

  mlir::Value transformedResult;
  if (cmpInfo.getOrdering() == CmpOrdering::Strong) {
    // Strong ordering.
    auto lt = buildCmp(CmpOpKind::lt);
    auto eq = buildCmp(CmpOpKind::eq);
    auto selectOnEq = buildSelect(eq, eqRes, gtRes);
    transformedResult = buildSelect(lt, ltRes, selectOnEq);
  } else {
    // Partial ordering.
    auto unorderedRes = buildCmpRes(cmpInfo.getUnordered().value());

    auto lt = buildCmp(CmpOpKind::lt);
    auto eq = buildCmp(CmpOpKind::eq);
    auto gt = buildCmp(CmpOpKind::gt);
    auto selectOnEq = buildSelect(eq, eqRes, unorderedRes);
    auto selectOnGt = buildSelect(gt, gtRes, selectOnEq);
    transformedResult = buildSelect(lt, ltRes, selectOnGt);
  }

  op.replaceAllUsesWith(transformedResult);
  op.erase();
}

void LoweringPreparePass::handleGlobalOpCtorDtor(GlobalOp globalOp) {
  auto &ctorRegion = globalOp.getCtorRegion();
  auto &dtorRegion = globalOp.getDtorRegion();

  if (!ctorRegion.empty() || !dtorRegion.empty()) {
    // Build a variable initialization function and move the initialzation code
    // in the ctor region over.
    auto f = buildCXXGlobalVarDeclInitFunc(globalOp);

    // Clear the ctor and dtor region
    ctorRegion.getBlocks().clear();
    dtorRegion.getBlocks().clear();

    // Add a function call to the variable initialization function.
    assert(!hasAttr<clang::InitPriorityAttr>(
               mlir::cast<ASTDeclInterface>(*globalOp.getAst())) &&
           "custom initialization priority NYI");
    dynamicInitializers.push_back(f);
  }

  std::optional<mlir::ArrayAttr> annotations = globalOp.getAnnotations();
  if (annotations)
    addGlobalAnnotations(globalOp, annotations.value());
}

void LoweringPreparePass::lowerGetGlobalOp(GetGlobalOp getGlobalOp) {
  if (!getGlobalOp.getStaticLocal())
    return;

  auto globalOp = mlir::cast<cir::GlobalOp>(
      mlir::SymbolTable::lookupSymbolIn(theModule, getGlobalOp.getName()));

  handleStaticLocal(globalOp, getGlobalOp);
}

void LoweringPreparePass::lowerGlobalOp(GlobalOp globalOp) {
  if (!globalOp.getStaticLocal())
    handleGlobalOpCtorDtor(globalOp);
}

static cir::GlobalOp createGuardGlobalOp(::cir::CIRBaseBuilderTy &builder,
                                         mlir::Location loc, StringRef name,
                                         mlir::Type type, bool isConstant,
                                         cir::AddressSpaceAttr addrSpace,
                                         cir::GlobalLinkageKind linkage,
                                         cir::FuncOp curFn) {

  cir::GlobalOp g;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // insert before the Fn requiring the guard var here
    builder.setInsertionPoint(curFn);

    g = builder.create<cir::GlobalOp>(loc, name, type, isConstant, linkage,
                                      addrSpace);

    // Default to private until we can judge based on the initializer,
    // since MLIR doesn't allow public declarations.
    mlir::SymbolTable::setSymbolVisibility(
        g, mlir::SymbolTable::Visibility::Private);
  }
  return g;
}

static mlir::Operation *getGlobalValue(mlir::ModuleOp theModule,
                                       StringRef name) {
  auto *global = mlir::SymbolTable::lookupSymbolIn(theModule, name);
  if (!global)
    return {};
  return global;
}

static cir::FuncOp createCIRFunction(clang::CIRGen::CIRGenBuilderTy &builder,
                                     mlir::MLIRContext &mlirContext,
                                     mlir::ModuleOp theModule,
                                     mlir::Location loc, StringRef name,
                                     cir::FuncType type) {
  // At the point we need to create the function, the insertion point
  // could be anywhere (e.g. callsite). Do not rely on whatever it might
  // be, properly save, find the appropriate place and restore.
  FuncOp f;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Get the first function in the module as the location to insert the new
    // function.
    Operation *firstFn = &theModule->getRegion(0).getBlocks().front().front();
    builder.setInsertionPoint(firstFn);

    f = builder.create<cir::FuncOp>(loc, name, type);

    assert(f.isDeclaration() && "expected empty body");

    // A declaration gets private visibility by default, but external linkage
    // as the default linkage.
    f.setLinkageAttr(cir::GlobalLinkageKindAttr::get(
        &mlirContext, cir::GlobalLinkageKind::ExternalLinkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);

    // Initialize with empty dict of extra attributes.
    f.setExtraAttrsAttr(cir::ExtraFuncAttributesAttr::get(
        &mlirContext, builder.getDictionaryAttr({})));
  }
  return f;
}

/// If the specified mangled name is not in the module,
/// create and return a CIR Function with the specified type. If there is
/// something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that corresponded to this. This is
/// used to set the attributes on the function when it is first created.
static cir::FuncOp getOrCreateCIRFunctionForRuntimeFunction(
    clang::ASTContext &astContext, mlir::MLIRContext &mlirContext,
    mlir::ModuleOp theModule, clang::CIRGen::CIRGenBuilderTy &builder,
    StringRef mangledName, mlir::Type type) {
  // Lookup the entry, lazily creating it if necessary.
  mlir::Operation *entry = getGlobalValue(theModule, mangledName);
  if (entry) {
    assert(isa<cir::FuncOp>(entry) &&
           "not implemented, only supports FuncOp for now");

    // If there are two attempts to define the same mangled name, issue an
    // error.
    auto fn = cast<cir::FuncOp>(entry);

    if (fn && fn.getFunctionType() == type) {
      return fn;
    }
  }

  // This function doesn't have a complete type (for example, the return type is
  // an incomplete struct). Use a fake type instead, and make sure not to try to
  // set attributes.
  bool isIncompleteFunction = false;

  cir::FuncType fTy;
  if (mlir::isa<cir::FuncType>(type)) {
    fTy = mlir::cast<cir::FuncType>(type);
  } else {
    assert(false && "NYI");
    // FTy = mlir::FunctionType::get(VoidTy, false);
    isIncompleteFunction = true;
  }

  // TODO: CodeGen includeds the linkage (ExternalLinkage) and only passes the
  // mangledname if Entry is nullptr
  auto func = createCIRFunction(builder, mlirContext, theModule,
                                theModule.getLoc(), mangledName, fTy);

  // If we already created a function with the same mangled name (but different
  // type) before, take its name and add it to the list of functions to be
  // replaced with F at the end of CodeGen.
  //
  // This happens if there is a prototype for a function (e.g. "int f()") and
  // then a definition of a different type (e.g. "int f(int x)").
  if (entry) {
    // Fetch a generic symbol-defining operation and its uses.
    auto symbolOp = dyn_cast<mlir::SymbolOpInterface>(entry);
    assert(symbolOp && "Expected a symbol-defining operation");

    // TODO(cir): When can this symbol be something other than a function?
    assert(isa<cir::FuncOp>(entry) && "NYI");

    // Obliterate no-proto declaration.
    entry->erase();
  }

  if (!isIncompleteFunction) {
    assert(func.getFunctionType() == type);
    return func;
  }

  // TODO(cir): Might need bitcast to different address space.
  assert(!cir::MissingFeatures::addressSpace());
  return func;
}

static cir::FuncOp createRuntimeFunction(
    clang::ASTContext &astContext, mlir::MLIRContext &mlirContext,
    mlir::ModuleOp theModule, clang::CIRGen::CIRGenBuilderTy &builder,

    cir::FuncType type, StringRef name, mlir::ArrayAttr = {},
    [[maybe_unused]] bool local = false, bool assumeConvergent = false) {
  if (assumeConvergent) {
    llvm_unreachable("NYI");
  }
  if (local)
    llvm_unreachable("NYI");

  auto entry = getOrCreateCIRFunctionForRuntimeFunction(
      astContext, mlirContext, theModule, builder, name, type);

  // Traditional codegen checks for a valid dyn_cast llvm::Function for `entry`,
  // no testcase that cover this path just yet though.
  if (!entry) {
    // Setup runtime CC, DLL support for windows and set dso local.
    llvm_unreachable("NYI");
  }

  return entry;
}

static cir::FuncOp getGuardAbortFn(clang::ASTContext &astContext,
                                   mlir::MLIRContext &mlirContext,
                                   mlir::ModuleOp theModule,
                                   clang::CIRGen::CIRGenBuilderTy &builder,
                                   cir::PointerType guardPtrTy) {
  // void __cxa_guard_abort(__guard *guard_object);
  cir::FuncType fTy = builder.getFuncType(guardPtrTy, {builder.getVoidTy()},
                                          /*isVarArg=*/false);
  assert(!cir::MissingFeatures::functionIndexAttribute());
  assert(!cir::MissingFeatures::noUnwindAttribute());
  return createRuntimeFunction(astContext, mlirContext, theModule, builder, fTy,
                               "__cxa_guard_abort");
}

static cir::FuncOp getGuardAcquireFn(clang::ASTContext &astContext,
                                     mlir::MLIRContext &mlirContext,
                                     mlir::ModuleOp theModule,
                                     clang::CIRGen::CIRGenBuilderTy &builder,
                                     cir::PointerType guardPtrTy) {
  // int __cxa_guard_acquire(__guard *guard_object);
  // TODO(CIR): The hardcoded getSInt32Ty is wrong here. CodeGen uses
  // CodeGenTypes.convertType but we don't have access to the CGM.
  cir::FuncType fTy = builder.getFuncType(guardPtrTy, {builder.getSInt32Ty()},
                                          /*isVarArg=*/false);
  assert(!cir::MissingFeatures::functionIndexAttribute());
  assert(!cir::MissingFeatures::noUnwindAttribute());
  return createRuntimeFunction(astContext, mlirContext, theModule, builder, fTy,
                               "__cxa_guard_acquire");
}

static cir::FuncOp getGuardReleaseFn(clang::ASTContext &astContext,
                                     mlir::MLIRContext &mlirContext,
                                     mlir::ModuleOp theModule,
                                     clang::CIRGen::CIRGenBuilderTy &builder,
                                     cir::PointerType guardPtrTy) {
  // void __cxa_guard_release(__guard *guard_object);
  cir::FuncType fTy = builder.getFuncType(guardPtrTy, {builder.getVoidTy()},
                                          /*isVarArg=*/false);
  assert(!cir::MissingFeatures::functionIndexAttribute());
  assert(!cir::MissingFeatures::noUnwindAttribute());
  return createRuntimeFunction(astContext, mlirContext, theModule, builder, fTy,
                               "__cxa_guard_release");
}

static mlir::Value emitRuntimeCall(clang::CIRGen::CIRGenBuilderTy &builder,
                                   mlir::Location loc, cir::FuncOp callee,
                                   ArrayRef<mlir::Value> args) {
  // TODO(cir): set the calling convention to this runtime call.
  assert(!cir::MissingFeatures::setCallingConv());

  auto call = builder.createCallOp(loc, callee, args);
  assert(call->getNumResults() <= 1 &&
         "runtime functions have at most 1 result");

  if (call->getNumResults() == 0)
    return nullptr;

  return call->getResult(0);
}

static mlir::Value
emitNounwindRuntimeCall(clang::CIRGen::CIRGenBuilderTy &builder,
                        mlir::Location loc, cir::FuncOp callee,
                        ArrayRef<mlir::Value> args) {
  mlir::Value call = emitRuntimeCall(builder, loc, callee, args);
  assert(!cir::MissingFeatures::noUnwindAttribute());
  return call;
}

void LoweringPreparePass::handleStaticLocal(GlobalOp globalOp,
                                            GetGlobalOp getGlobalOp) {

  std::optional<cir::ASTVarDeclInterface> astOption = globalOp.getAst();
  assert(astOption.has_value());
  cir::ASTVarDeclInterface varDecl = astOption.value();

  builder->setInsertionPointAfter(getGlobalOp);
  Block *getGlobalOpBlock = builder->getInsertionBlock();
  // TODO(CIR): This is too simple at the moment. This is only tested on a
  // simple test case with only the static local var decl and thus we only have
  // the return. For less trivial examples we'll have to handle shuffling the
  // contents of this block more carefully.
  Operation *ret = getGlobalOpBlock->getTerminator();
  ret->remove();
  builder->setInsertionPointAfter(getGlobalOp);

  // Inline variables that weren't instantiated from variable templates have
  // partially-ordered initialization within their translation unit.
  bool nonTemplateInline =
      varDecl.isInline() &&
      !isTemplateInstantiation(varDecl.getTemplateSpecializationKind());

  // We only need to use thread-safe statics for local non-TLS variables and
  // inline variables; other global initialization is always single-threaded
  // or (through lazy dynamic loading in multiple threads) unsequenced.
  bool threadsafe = astCtx->getLangOpts().ThreadsafeStatics &&
                    (varDecl.isLocalVarDecl() || nonTemplateInline) &&
                    !varDecl.getTLSKind();

  // If we have a global variable with internal linkage and thread-safe
  // statics are disabled, we can just let the guard variable be of type i8.
  bool useInt8GuardVariable = !threadsafe && globalOp.hasInternalLinkage();

  cir::IntType guardTy;
  clang::CharUnits guardAlignment;
  if (useInt8GuardVariable) {
    guardTy = cir::IntType::get(&getContext(), 8, /*isSigned=*/true);
    guardAlignment = clang::CharUnits::One();
  } else {
    // Guard variables are 64 bits in the generic ABI and size width on ARM
    // (i.e. 32-bit on AArch32, 64-bit on AArch64).
    if (::cir::MissingFeatures::useARMGuardVarABI()) {
      llvm_unreachable("NYI");
    } else {
      guardTy = cir::IntType::get(&getContext(), 64, /*isSigned=*/true);
      cir::CIRDataLayout dataLayout(theModule);
      guardAlignment =
          clang::CharUnits::fromQuantity(dataLayout.getABITypeAlign(guardTy));
    }
  }
  auto guardPtrTy = cir::PointerType::get(guardTy);

  // Create the guard variable if we don't already have it (as we might if
  // we're double-emitting this function body).
  cir::GlobalOp guard = getStaticLocalDeclGuardAddress(varDecl);
  if (!guard) {
    // Mangle the name for the guard.
    SmallString<256> guardName;
    {
      llvm::raw_svector_ostream out(guardName);
      varDecl.mangleStaticGuardVariable(out);
    }

    // Create the guard variable with a zero-initializer.
    // Just absorb linkage, visibility and dll storage class from the guarded
    // variable.
    guard = createGuardGlobalOp(
        *builder, globalOp->getLoc(), guardName, guardTy,
        /*isConstant=*/false, /*addrSpace=*/{}, globalOp.getLinkage(),
        getGlobalOp->getParentOfType<cir::FuncOp>());
    guard.setInitialValueAttr(cir::IntAttr::get(guardTy, 0));
    guard.setDSOLocal(globalOp.isDSOLocal());
    guard.setVisibility(globalOp.getVisibility());
    assert(!::cir::MissingFeatures::setDLLStorageClass());
    // guard.setDLLStorageClass(globalOp.getDLLStorageClass());
    // If the variable is thread-local, so is its guard variable.
    assert(!::cir::MissingFeatures::threadLocal());
    // guard.setThreadLocalMode(globalOp.getThreadLocalMode());
    guard.setAlignment(guardAlignment.getAsAlign().value());

    // The ABI says: "It is suggested that it be emitted in the same COMDAT
    // group as the associated data object." In practice, this doesn't work
    // for non-ELF and non-Wasm object formats, so only do it for ELF and
    // Wasm.
    assert(!::cir::MissingFeatures::setComdat());

    setStaticLocalDeclGuardAddress(varDecl, guard);
  }

  mlir::Value getGuard = builder->createGetGlobal(guard, /*threadLocal*/ false);
  clang::CIRGen::Address guardAddr =
      clang::CIRGen::Address(getGuard, guard.getSymType(), guardAlignment);

  // Test whether the variable has completed initialization.
  //
  // Itanium C++ ABI 3.3.2:
  //   The following is pseudo-code showing how these functions can be used:
  //     if (obj_guard.first_byte == 0) {
  //       if ( __cxa_guard_acquire (&obj_guard) ) {
  //         try {
  //           ... initialize the object ...;
  //         } catch (...) {
  //           __cxa_guard_abort (&obj_guard);
  //           throw;
  //         }
  //         ... queue object destructor with __cxa_atexit() ...;
  //         __cxa_guard_release (&obj_guard);
  //       }
  //     }
  //
  // If threadsafe statics are enabled, but we don't have inline atomics, just
  // call __cxa_guard_acquire unconditionally.  The "inline" check isn't
  // actually inline, and the user might not expect calls to __atomic
  // libcalls.
  unsigned maxInlineWidthInbits =
      astCtx->getTargetInfo().getMaxAtomicInlineWidth();

  auto initBlock = [&]() {
    // CIR: Move the initializer from the globalOp's ctor region into the
    // current block.
    // TODO(CIR): Once we support exceptions we'll need to walk the ctor region
    // to change calls to invokes.
    auto &ctorRegion = globalOp.getCtorRegion();
    assert(!ctorRegion.empty() && "This should never be empty here.");
    if (!ctorRegion.hasOneBlock())
      llvm_unreachable("Multiple blocks NYI");
    Block &block = ctorRegion.front();
    Block *insertBlock = builder->getInsertionBlock();
    insertBlock->getOperations().splice(insertBlock->end(),
                                        block.getOperations(), block.begin(),
                                        std::prev(block.end()));
    builder->setInsertionPointToEnd(insertBlock);

    ctorRegion.getBlocks().clear();

    if (threadsafe) {
      // NOTE(CIR): CodeGen clears the above pushed CallGuardAbort here and thus
      // the __guard_abort gets inserted. We'll have to figure out how to
      // properly handle this when supporting static locals with exceptions.

      // Call __cxa_guard_release. This cannot throw.
      emitNounwindRuntimeCall(*builder, globalOp->getLoc(),
                              getGuardReleaseFn(*astCtx, getContext(),
                                                theModule, *builder,
                                                guardPtrTy),
                              guardAddr.emitRawPointer());
    } else if (varDecl.isLocalVarDecl()) {
      llvm_unreachable("NYI");
    }
  };

  // The semantics of dynamic initialization of variables with static or
  // thread storage duration depends on whether they are declared at
  // block-scope. The initialization of such variables at block-scope can be
  // aborted with an exception and later retried (per C++20 [stmt.dcl]p4), and
  // recursive entry to their initialization has undefined behavior (also per
  // C++20 [stmt.dcl]p4). For such variables declared at non-block scope,
  // exceptions lead to termination (per C++20 [except.terminate]p1), and
  // recursive references to the variables are governed only by the lifetime
  // rules (per C++20 [class.cdtor]p2), which means such references are
  // perfectly fine as long as they avoid touching memory. As a result,
  // block-scope variables must not be marked as initialized until after
  // initialization completes (unless the mark is reverted following an
  // exception), but non-block-scope variables must be marked prior to
  // initialization so that recursive accesses during initialization do not
  // restart initialization.

  // Variables used when coping with thread-safe statics and exceptions.
  auto guardAcquireBlock = [&]() {
    if (threadsafe) {
      auto loc = globalOp->getLoc();
      // Call __cxa_guard_acquire.
      mlir::Value value = emitNounwindRuntimeCall(
          *builder, loc,
          getGuardAcquireFn(*astCtx, getContext(), theModule, *builder,
                            guardPtrTy),
          guardAddr.emitRawPointer());

      auto isNotNull = builder->createIsNotNull(loc, value);
      builder->create<cir::IfOp>(globalOp.getLoc(), isNotNull,
                                 /*=withElseRegion*/ false,
                                 [&](mlir::OpBuilder &, mlir::Location) {
                                   initBlock();
                                   builder->createYield(getGlobalOp->getLoc());
                                 });

      // NOTE(CIR): CodeGen pushes a CallGuardAbort cleanup here, but we are
      // synthesizing the outcome via walking the CIR in the ctor region and
      // changing calls to invokes.

    } else if (!varDecl.isLocalVarDecl()) {
      llvm_unreachable("NYI");
    }
  };

  if (!threadsafe || maxInlineWidthInbits) {
    // Load the first byte of the guard variable.
    mlir::Value load = builder->createAlignedLoad(
        getGlobalOp.getLoc(), builder->getSInt8Ty(), guardAddr.getPointer(),
        guardAddr.getAlignment());

    // Itanium ABI:
    //   An implementation supporting thread-safety on multiprocesor systems
    //   must also guarantee that references to the initialized object do not
    //   occur before the load of the initialization flag.
    //
    // In LLVM, we do this by marking the load Acquire.
    if (threadsafe)
      cast<cir::LoadOp>(load.getDefiningOp()).setAtomic(cir::MemOrder::Acquire);

    // For ARM, we should only check the first bit, rather than the entire
    // byte:
    //
    // ARM C++ ABI 3.2.3.1:
    //   To support the potential use of initialization guard variables
    //   as semaphores that are the target of ARM SWP and LDREX/STREX
    //   synchronizing instructions we define a static initialization
    //   guard variable to be a 4-byte aligned, 4-byte word with the
    //   following inline access protocol.
    //     #define INITIALIZED 1
    //     if ((obj_guard & INITIALIZED) != INITIALIZED) {
    //       if (__cxa_guard_acquire(&obj_guard))
    //         ...
    //     }
    //
    // and similarly for ARM64:
    //
    // ARM64 C++ ABI 3.2.2:
    //   This ABI instead only specifies the value bit 0 of the static guard
    //   variable; all other bits are platform defined. Bit 0 shall be 0 when
    //   the variable is not initialized and 1 when it is.
    if (MissingFeatures::useARMGuardVarABI() && !useInt8GuardVariable)
      llvm_unreachable("NYI");
    mlir::Value constOne = builder->getConstInt(
        getGlobalOp->getLoc(), llvm::APSInt(llvm::APInt(8, 1),
                                            /*isUnsigned=*/false));
    mlir::Value value =
        (!cir::MissingFeatures::useARMGuardVarABI() && !useInt8GuardVariable)
            ? builder->createAnd(load, constOne)
            : load;
    mlir::Value needsInit =
        builder->createIsNull(globalOp.getLoc(), value, "guard.uninitialized");

    builder->create<cir::IfOp>(globalOp.getLoc(), needsInit,
                               /*=withElseRegion*/ false,
                               [&](mlir::OpBuilder &, mlir::Location) {
                                 if (MissingFeatures::metaDataNode())
                                   llvm_unreachable("NYI");
                                 guardAcquireBlock();
                                 builder->createYield(getGlobalOp->getLoc());
                               });
  }

  builder->setInsertionPointToEnd(getGlobalOpBlock);
  builder->insert(ret);
}

void LoweringPreparePass::buildGlobalCtorDtorList() {
  if (!globalCtorList.empty()) {
    theModule->setAttr(cir::CIRDialect::getGlobalCtorsAttrName(),
                       mlir::ArrayAttr::get(&getContext(), globalCtorList));
  }
  if (!globalDtorList.empty()) {
    theModule->setAttr(cir::CIRDialect::getGlobalDtorsAttrName(),
                       mlir::ArrayAttr::get(&getContext(), globalDtorList));
  }
}

void LoweringPreparePass::buildCXXGlobalInitFunc() {
  if (dynamicInitializers.empty())
    return;

  for (auto &f : dynamicInitializers) {
    // TODO: handle globals with a user-specified initialzation priority.
    auto ctorAttr = cir::GlobalCtorAttr::get(&getContext(), f.getName());
    globalCtorList.push_back(ctorAttr);
  }

  SmallString<256> fnName;
  // Include the filename in the symbol name. Including "sub_" matches gcc
  // and makes sure these symbols appear lexicographically behind the symbols
  // with priority emitted above.  Module implementation units behave the same
  // way as a non-modular TU with imports.
  // TODO: check CXX20ModuleInits
  if (astCtx->getCurrentNamedModule() &&
      !astCtx->getCurrentNamedModule()->isModuleImplementation()) {
    llvm::raw_svector_ostream Out(fnName);
    std::unique_ptr<clang::MangleContext> MangleCtx(
        astCtx->createMangleContext());
    cast<clang::ItaniumMangleContext>(*MangleCtx)
        .mangleModuleInitializer(astCtx->getCurrentNamedModule(), Out);
  } else {
    fnName += "_GLOBAL__sub_I_";
    fnName += getTransformedFileName(theModule);
  }

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToEnd(&theModule.getBodyRegion().back());
  auto fnType =
      cir::FuncType::get({}, cir::VoidType::get(builder.getContext()));
  FuncOp f = buildRuntimeFunction(builder, fnName, theModule.getLoc(), fnType,
                                  cir::GlobalLinkageKind::ExternalLinkage);
  builder.setInsertionPointToStart(f.addEntryBlock());
  for (auto &f : dynamicInitializers) {
    builder.createCallOp(f.getLoc(), f);
  }

  builder.create<ReturnOp>(f.getLoc());
}

void LoweringPreparePass::lowerDynamicCastOp(DynamicCastOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op);

  assert(astCtx && "AST context is not available during lowering prepare");
  auto loweredValue = cxxABI->lowerDynamicCast(builder, *astCtx, op);

  op.replaceAllUsesWith(loweredValue);
  op.erase();
}

static void lowerArrayDtorCtorIntoLoop(CIRBaseBuilderTy &builder,
                                       mlir::Operation *op, mlir::Type eltTy,
                                       mlir::Value arrayAddr,
                                       uint64_t arrayLen) {
  // Generate loop to call into ctor/dtor for every element.
  auto loc = op->getLoc();

  // TODO: instead of fixed integer size, create alias for PtrDiffTy and unify
  // with CIRGen stuff.
  auto ptrDiffTy =
      cir::IntType::get(builder.getContext(), 64, /*signed=*/false);
  auto numArrayElementsConst = builder.create<cir::ConstantOp>(
      loc, ptrDiffTy, cir::IntAttr::get(ptrDiffTy, arrayLen));

  auto begin = builder.create<cir::CastOp>(
      loc, eltTy, cir::CastKind::array_to_ptrdecay, arrayAddr);
  mlir::Value end = builder.create<cir::PtrStrideOp>(loc, eltTy, begin,
                                                     numArrayElementsConst);

  auto tmpAddr = builder.createAlloca(
      loc, /*addr type*/ builder.getPointerTo(eltTy),
      /*var type*/ eltTy, "__array_idx", clang::CharUnits::One());
  builder.createStore(loc, begin, tmpAddr);

  auto loop = builder.createDoWhile(
      loc,
      /*condBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto currentElement = b.create<cir::LoadOp>(loc, eltTy, tmpAddr);
        mlir::Type boolTy = cir::BoolType::get(b.getContext());
        auto cmp = builder.create<cir::CmpOp>(loc, boolTy, cir::CmpOpKind::eq,
                                              currentElement, end);
        builder.createCondition(cmp);
      },
      /*bodyBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        auto currentElement = b.create<cir::LoadOp>(loc, eltTy, tmpAddr);

        CallOp ctorCall;
        op->walk([&](CallOp c) { ctorCall = c; });
        assert(ctorCall && "expected ctor call");

        auto one = builder.create<cir::ConstantOp>(
            loc, ptrDiffTy, cir::IntAttr::get(ptrDiffTy, 1));

        ctorCall->moveAfter(one);
        ctorCall->setOperand(0, currentElement);

        // Advance pointer and store them to temporary variable
        auto nextElement =
            builder.create<cir::PtrStrideOp>(loc, eltTy, currentElement, one);
        builder.createStore(loc, nextElement, tmpAddr);
        builder.createYield(loc);
      });

  op->replaceAllUsesWith(loop);
  op->erase();
}

void LoweringPreparePass::lowerArrayDtor(ArrayDtor op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  auto eltTy = op->getRegion(0).getArgument(0).getType();
  auto arrayLen =
      mlir::cast<cir::ArrayType>(
          mlir::cast<cir::PointerType>(op.getAddr().getType()).getPointee())
          .getSize();
  lowerArrayDtorCtorIntoLoop(builder, op, eltTy, op.getAddr(), arrayLen);
}

static std::string getGlobalVarNameForConstString(cir::StoreOp op,
                                                  uint64_t &cnt) {
  llvm::SmallString<64> finalName;
  llvm::raw_svector_ostream Out(finalName);

  Out << "__const.";
  if (auto fnOp = op->getParentOfType<cir::FuncOp>()) {
    Out << fnOp.getSymNameAttr().getValue() << ".";
  } else {
    Out << "module.";
  }

  auto allocaOp = dyn_cast_or_null<cir::AllocaOp>(op.getAddr().getDefiningOp());
  if (allocaOp && !allocaOp.getName().empty())
    Out << allocaOp.getName();
  else
    Out << cnt++;
  return finalName.c_str();
}

void LoweringPreparePass::lowerToMemCpy(StoreOp op) {
  // Now that basic filter is done, do more checks before proceding with the
  // transformation.
  auto cstOp =
      dyn_cast_if_present<cir::ConstantOp>(op.getValue().getDefiningOp());
  if (!cstOp)
    return;

  if (!isa<cir::ConstArrayAttr>(cstOp.getValue()))
    return;
  CIRBaseBuilderTy builder(getContext());

  // Create a global which is initialized with the attribute that is either a
  // constant array or struct.
  assert(!cir::MissingFeatures::unnamedAddr() && "NYI");
  builder.setInsertionPointToStart(&theModule.getBodyRegion().front());
  std::string globalName =
      getGlobalVarNameForConstString(op, annonGlobalConstArrayCount);
  cir::GlobalOp globalCst = buildRuntimeVariable(
      builder, globalName, op.getLoc(), op.getValue().getType(),
      cir::GlobalLinkageKind::PrivateLinkage);
  globalCst.setInitialValueAttr(cstOp.getValue());
  globalCst.setConstant(true);

  // Transform the store into a cir.copy.
  builder.setInsertionPointAfter(op.getOperation());
  cir::CopyOp memCpy =
      builder.createCopy(op.getAddr(), builder.createGetGlobal(globalCst));
  op->replaceAllUsesWith(memCpy);
  op->erase();
  if (cstOp->getResult(0).getUsers().empty())
    cstOp->erase();
}

void LoweringPreparePass::lowerArrayCtor(ArrayCtor op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());

  auto eltTy = op->getRegion(0).getArgument(0).getType();
  auto arrayLen =
      mlir::cast<cir::ArrayType>(
          mlir::cast<cir::PointerType>(op.getAddr().getType()).getPointee())
          .getSize();
  lowerArrayDtorCtorIntoLoop(builder, op, eltTy, op.getAddr(), arrayLen);
}

void LoweringPreparePass::lowerStdFindOp(StdFindOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());
  auto call = builder.createCallOp(
      op.getLoc(), op.getOriginalFnAttr(), op.getResult().getType(),
      mlir::ValueRange{op.getOperand(0), op.getOperand(1), op.getOperand(2)});

  op.replaceAllUsesWith(call);
  op.erase();
}

void LoweringPreparePass::lowerIterBeginOp(IterBeginOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());
  auto call = builder.createCallOp(op.getLoc(), op.getOriginalFnAttr(),
                                   op.getResult().getType(),
                                   mlir::ValueRange{op.getOperand()});

  op.replaceAllUsesWith(call);
  op.erase();
}

void LoweringPreparePass::lowerIterEndOp(IterEndOp op) {
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(op.getOperation());
  auto call = builder.createCallOp(op.getLoc(), op.getOriginalFnAttr(),
                                   op.getResult().getType(),
                                   mlir::ValueRange{op.getOperand()});

  op.replaceAllUsesWith(call);
  op.erase();
}

void LoweringPreparePass::addGlobalAnnotations(mlir::Operation *op,
                                               mlir::ArrayAttr annotations) {
  auto globalValue = cast<mlir::SymbolOpInterface>(op);
  mlir::StringAttr globalValueName = globalValue.getNameAttr();
  for (auto &annot : annotations) {
    llvm::SmallVector<mlir::Attribute, 2> entryArray = {globalValueName, annot};
    globalAnnotations.push_back(
        mlir::ArrayAttr::get(theModule.getContext(), entryArray));
  }
}

void LoweringPreparePass::buildGlobalAnnotationValues() {
  if (globalAnnotations.empty())
    return;
  mlir::ArrayAttr annotationValueArray =
      mlir::ArrayAttr::get(theModule.getContext(), globalAnnotations);
  theModule->setAttr(cir::CIRDialect::getGlobalAnnotationsAttrName(),
                     cir::GlobalAnnotationValuesAttr::get(
                         theModule.getContext(), annotationValueArray));
}

void LoweringPreparePass::runOnOp(Operation *op) {
  if (auto unary = dyn_cast<UnaryOp>(op)) {
    lowerUnaryOp(unary);
  } else if (auto bin = dyn_cast<BinOp>(op)) {
    lowerBinOp(bin);
  } else if (auto cast = dyn_cast<CastOp>(op)) {
    lowerCastOp(cast);
  } else if (auto complexBin = dyn_cast<ComplexBinOp>(op)) {
    lowerComplexBinOp(complexBin);
  } else if (auto threeWayCmp = dyn_cast<CmpThreeWayOp>(op)) {
    lowerThreeWayCmpOp(threeWayCmp);
  } else if (auto vaArgOp = dyn_cast<VAArgOp>(op)) {
    lowerVAArgOp(vaArgOp);
  } else if (auto global = dyn_cast<GlobalOp>(op)) {
    lowerGlobalOp(global);
  } else if (auto getGlobal = dyn_cast<GetGlobalOp>(op)) {
    lowerGetGlobalOp(getGlobal);
  } else if (auto dynamicCast = dyn_cast<DynamicCastOp>(op)) {
    lowerDynamicCastOp(dynamicCast);
  } else if (auto stdFind = dyn_cast<StdFindOp>(op)) {
    lowerStdFindOp(stdFind);
  } else if (auto iterBegin = dyn_cast<IterBeginOp>(op)) {
    lowerIterBeginOp(iterBegin);
  } else if (auto iterEnd = dyn_cast<IterEndOp>(op)) {
    lowerIterEndOp(iterEnd);
  } else if (auto arrayCtor = dyn_cast<ArrayCtor>(op)) {
    lowerArrayCtor(arrayCtor);
  } else if (auto arrayDtor = dyn_cast<ArrayDtor>(op)) {
    lowerArrayDtor(arrayDtor);
  } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
    mlir::Type valTy = storeOp.getValue().getType();
    if (isa<cir::ArrayType>(valTy) || isa<cir::StructType>(valTy))
      lowerToMemCpy(storeOp);
  } else if (auto fnOp = dyn_cast<cir::FuncOp>(op)) {
    if (auto globalCtor = fnOp.getGlobalCtorAttr()) {
      globalCtorList.push_back(globalCtor);
    } else if (auto globalDtor = fnOp.getGlobalDtorAttr()) {
      globalDtorList.push_back(globalDtor);
    }
    if (std::optional<mlir::ArrayAttr> annotations = fnOp.getAnnotations())
      addGlobalAnnotations(fnOp, annotations.value());
  }
}

void LoweringPreparePass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op)) {
    theModule = cast<::mlir::ModuleOp>(op);
  }

  llvm::SmallVector<Operation *> opsToTransform;

  op->walk([&](Operation *op) {
    if (isa<UnaryOp, BinOp, CastOp, ComplexBinOp, CmpThreeWayOp, VAArgOp,
            GetGlobalOp, GlobalOp, DynamicCastOp, StdFindOp, IterEndOp,
            IterBeginOp, ArrayCtor, ArrayDtor, cir::FuncOp, StoreOp>(op))
      opsToTransform.push_back(op);
  });

  for (auto *o : opsToTransform)
    runOnOp(o);

  buildCXXGlobalInitFunc();
  buildGlobalCtorDtorList();
  buildGlobalAnnotationValues();
}

std::unique_ptr<Pass> mlir::createLoweringPreparePass() {
  return std::make_unique<LoweringPreparePass>();
}
std::unique_ptr<Pass>
mlir::createLoweringPreparePass(clang::ASTContext *astCtx,
                                clang::CIRGen::CIRGenBuilderTy &builder) {
  auto pass = std::make_unique<LoweringPreparePass>();
  pass->setBuilder(builder);
  pass->setASTContext(astCtx);
  return std::move(pass);
}
