//===- IdiomRecognizer.cpp - Recognize and raise C/C++ library calls ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

#include "StdHelpers.h"

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace cir;

namespace {

// Recognizes a cir.call that calls a standard library function represented
// by `TargetOp`, and raise it to that operation.
template <typename TargetOp> class StdRecognizer {
private:
  // Reserved for template specialization.
  static bool checkArguments(mlir::ValueRange) { return true; }

  template <size_t... Indices>
  static TargetOp buildCall(CIRBaseBuilderTy &builder, CallOp call,
                            std::index_sequence<Indices...>) {
    return builder.create<TargetOp>(call.getLoc(), call.getResult().getType(),
                                    call.getCalleeAttr(),
                                    call.getOperand(Indices)...);
  }

public:
  static bool raise(CallOp call, mlir::MLIRContext &context, bool remark) {
    constexpr int numArgs = TargetOp::getNumArgs();
    if (call.getNumOperands() != numArgs)
      return false;

    auto callExprAttr = call.getAstAttr();
    llvm::StringRef stdFuncName = TargetOp::getFunctionName();
    if (!callExprAttr || !callExprAttr.isStdFunctionCall(stdFuncName))
      return false;

    if (!checkArguments(call.getArgOperands()))
      return false;

    if (remark)
      mlir::emitRemark(call.getLoc())
          << "found call to std::" << stdFuncName << "()";

    CIRBaseBuilderTy builder(context);
    builder.setInsertionPointAfter(call.getOperation());
    TargetOp op = buildCall(builder, call, std::make_index_sequence<numArgs>());
    call.replaceAllUsesWith(op);
    call.erase();
    return true;
  }
};

struct IdiomRecognizerPass : public IdiomRecognizerBase<IdiomRecognizerPass> {
  IdiomRecognizerPass() = default;
  void runOnOperation() override;
  void recognizeCall(CallOp call);
  bool raiseIteratorBeginEnd(CallOp call);

  // Handle pass options
  struct Options {
    enum : unsigned {
      None = 0,
      RemarkFoundCalls = 1,
      RemarkAll = 1 << 1,
    };
    unsigned val = None;
    bool isOptionsParsed = false;

    void parseOptions(ArrayRef<llvm::StringRef> remarks) {
      if (isOptionsParsed)
        return;

      for (auto &remark : remarks) {
        val |= StringSwitch<unsigned>(remark)
                   .Case("found-calls", RemarkFoundCalls)
                   .Case("all", RemarkAll)
                   .Default(None);
      }
      isOptionsParsed = true;
    }

    void parseOptions(IdiomRecognizerPass &pass) {
      llvm::SmallVector<llvm::StringRef, 4> remarks;

      for (auto &r : pass.remarksList)
        remarks.push_back(r);

      parseOptions(remarks);
    }

    bool emitRemarkAll() { return val & RemarkAll; }
    bool emitRemarkFoundCalls() {
      return emitRemarkAll() || val & RemarkFoundCalls;
    }
  } opts;

  ///
  /// AST related
  /// -----------
  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

static bool isIteratorLikeType(mlir::Type t) {
  // TODO: some iterators are going to be represented with structs,
  // in which case we could look at ASTRecordDeclInterface for more
  // information.
  auto pTy = dyn_cast<PointerType>(t);
  if (!pTy || !mlir::isa<cir::IntType>(pTy.getPointee()))
    return false;
  return true;
}

static bool isIteratorInStdContainter(mlir::Type t) {
  // TODO: only std::array supported for now, generalize and
  // use tablegen. CallDescription.cpp in the static analyzer
  // could be a good inspiration source too.
  return isStdArrayType(t);
}

bool IdiomRecognizerPass::raiseIteratorBeginEnd(CallOp call) {
  // FIXME: tablegen all of this function.
  CIRBaseBuilderTy builder(getContext());

  if (call.getNumOperands() != 1 || call.getNumResults() != 1)
    return false;

  auto callExprAttr = call.getAstAttr();
  if (!callExprAttr)
    return false;

  if (!isIteratorLikeType(call.getResult().getType()))
    return false;

  // First argument is the container "this" pointer.
  auto thisPtr = mlir::dyn_cast<cir::PointerType>(call.getOperand(0).getType());
  if (!thisPtr || !isIteratorInStdContainter(thisPtr.getPointee()))
    return false;

  builder.setInsertionPointAfter(call.getOperation());
  mlir::Operation *iterOp;
  if (callExprAttr.isIteratorBeginCall()) {
    if (opts.emitRemarkFoundCalls())
      emitRemark(call.getLoc()) << "found call to begin() iterator";
    iterOp = builder.create<cir::IterBeginOp>(
        call.getLoc(), call.getResult().getType(), call.getCalleeAttr(),
        call.getOperand(0));
  } else if (callExprAttr.isIteratorEndCall()) {
    if (opts.emitRemarkFoundCalls())
      emitRemark(call.getLoc()) << "found call to end() iterator";
    iterOp = builder.create<cir::IterEndOp>(
        call.getLoc(), call.getResult().getType(), call.getCalleeAttr(),
        call.getOperand(0));
  } else {
    return false;
  }

  call.replaceAllUsesWith(iterOp);
  call.erase();
  return true;
}

void IdiomRecognizerPass::recognizeCall(CallOp call) {
  if (raiseIteratorBeginEnd(call))
    return;

  bool remark = opts.emitRemarkFoundCalls();

  using StdFunctionsRecognizer = std::tuple<StdRecognizer<StdFindOp>>;

  // MSVC requires explicitly capturing these variables.
  std::apply(
      [&, call, remark, this](auto... recognizers) {
        (decltype(recognizers)::raise(call, this->getContext(), remark) || ...);
      },
      StdFunctionsRecognizer());
}

void IdiomRecognizerPass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  opts.parseOptions(*this);
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op))
    theModule = cast<::mlir::ModuleOp>(op);

  llvm::SmallVector<CallOp> callsToTransform;
  op->walk([&](CallOp callOp) {
    // Process call operations

    // Skip indirect calls.
    auto c = callOp.getCallee();
    if (!c)
      return;
    callsToTransform.push_back(callOp);
  });

  for (auto c : callsToTransform)
    recognizeCall(c);
}

std::unique_ptr<Pass> mlir::createIdiomRecognizerPass() {
  return std::make_unique<IdiomRecognizerPass>();
}

std::unique_ptr<Pass>
mlir::createIdiomRecognizerPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<IdiomRecognizerPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
