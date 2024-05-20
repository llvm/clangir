//===- llvm/DataLayout.h - Data size & alignment info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics llvm/IR/DataLayout.h. The queries are adapted to
// operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_DATALAYOUT_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_DATALAYOUT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

using Align = llvm::Align;
using Error = llvm::Error;

namespace mlir {
namespace cir {

/// A parsed version of the target data layout string in and methods for
/// querying it.
///
/// The target data layout string is specified *by the target* - a frontend
/// generating CIR is required to generate the right target data for the target
/// being lowered to.
class CIRDataLayout {
  // FIXME(cir): This class should use the existing MLIR data layout infra. It
  // could be replaced by a CIRDataLayout interface that wraps the basic MLIR
  // data layout interface providing more functionalities.

  DataLayout DL; // MLIR's DataLayout.

public:
  /// Constructs a DataLayout from a specification string. See reset().
  explicit CIRDataLayout(StringRef dataLayout, ModuleOp module) : DL(module) {
    reset(dataLayout);
  }

  /// Parse a data layout string (with fallback to default values).
  void reset(StringRef dataLayout);

  // Free all internal data structures.
  void clear();
};

} // end namespace cir
} // end namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_DATALAYOUT_H
