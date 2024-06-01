//===- CIRQueries.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/ASTContext.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRQueries.h"
#include "MissingFeature.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>

namespace mlir {
namespace cir {

CIRQueries::CIRQueries(MLIRContext *MLIRCtx, clang::LangOptions &LOpts)
    : MLIRCtx(MLIRCtx), LangOpts(LOpts) {}

CIRQueries::~CIRQueries() {}

Type CIRQueries::initBuiltinType(clang::BuiltinType::Kind K) {
  Type Ty;

  // NOTE(cir): Clang does more stuff here. Not sure if we need to do the same.
  assert(MissingFeature::qualifiedTypes());
  switch (K) {
  case clang::BuiltinType::Char_S:
    Ty = IntType::get(getMLIRContext(), 8, true);
    break;
  default:
    llvm_unreachable("NYI");
  }

  Types.push_back(Ty);
  return Ty;
}

void CIRQueries::initBuiltinTypes(const clang::TargetInfo &Target,
                                  const clang::TargetInfo *AuxTarget) {
  assert((!this->Target || this->Target == &Target) &&
         "Incorrect target reinitialization");
  this->Target = &Target;
  this->AuxTarget = AuxTarget;

  // C99 6.2.5p3.
  if (LangOpts.CharIsSigned)
    CharTy = initBuiltinType(clang::BuiltinType::Char_S);
  else
    llvm_unreachable("NYI");
}

} // namespace cir
} // namespace mlir
