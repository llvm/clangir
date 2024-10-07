//===--- ABIInfoImpl.cpp - Encapsulate calling convention details ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/ABIInfoImpl.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "ABIInfo.h"
#include "CIRCXXABI.h"
#include "LowerFunction.h"
#include "LowerFunctionInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &cxxabi, LowerFunctionInfo &fi,
                        const ABIInfo &info) {
  Type ty = fi.getReturnType();

  if (const auto rt = dyn_cast<StructType>(ty)) {
    assert(!::cir::MissingFeatures::isCXXRecordDecl());
  }

  return cxxabi.classifyReturnType(fi);
}

bool isAggregateTypeForABI(Type t) {
  assert(!::cir::MissingFeatures::functionMemberPointerType());
  return !LowerFunction::hasScalarEvaluationKind(t);
}

Type useFirstFieldIfTransparentUnion(Type ty) {
  if (auto rt = dyn_cast<StructType>(ty)) {
    if (rt.isUnion())
      llvm_unreachable("NYI");
  }
  return ty;
}

CIRCXXABI::RecordArgABI getRecordArgABI(const StructType rt,
                                        CIRCXXABI &cxxabi) {
  if (::cir::MissingFeatures::typeIsCXXRecordDecl()) {
    llvm_unreachable("NYI");
  }
  return cxxabi.getRecordArgABI(rt);
}

} // namespace cir
} // namespace mlir
