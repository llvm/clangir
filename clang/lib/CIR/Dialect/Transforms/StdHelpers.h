//===- StdHelpers.h - Helpers for standard types/functions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
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

#ifndef DIALECT_CIR_TRANSFORMS_STDHELPERS_H_
#define DIALECT_CIR_TRANSFORMS_STDHELPERS_H_

namespace cir {

bool isStdArrayType(mlir::Type t);

// Recognizes a standard function represented by `StdFuncID` with arguments
// count equal to `NumArgs`, and raise it to an instance of `TargetOp`. Using
// IDs is a workaround, as we can't pass string literals to template arguments
// in C++17.
template <int NumArgs, typename TargetOp, StdFuncID ID> class StdRecognizer {
private:
  // Reserved for template specialization.
  static bool checkArguments(mlir::ValueRange) { return true; }

public:
  static bool raise(CallOp call, mlir::MLIRContext &context, bool remark) {
    if (call.getNumOperands() != NumArgs)
      return false;

    auto callExprAttr = call.getAstAttr();
    llvm::StringRef stdFuncName = stringifyStdFuncID(ID);
    if (!callExprAttr || !callExprAttr.isStdFunctionCall(stdFuncName))
      return false;

    if (!checkArguments(call.getArgOperands()))
      return false;

    if (remark)
      mlir::emitRemark(call.getLoc())
          << "found call to std::" << stdFuncName << "()";

    CIRBaseBuilderTy builder(context);
    builder.setInsertionPointAfter(call.getOperation());
    TargetOp op =
        builder.create<TargetOp>(call.getLoc(), call.getResult().getType(),
                                 call.getCalleeAttr(), call.getOperands());
    call.replaceAllUsesWith(op);
    call.erase();
    return true;
  }
};

} // namespace cir

#endif
