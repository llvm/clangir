//===----- CIRCXXABI.h - Interface to C++ ABIs for CIR Dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the CodeGen/CGCXXABI.h class. The main difference
// is that this is adapted to operate on the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H

namespace mlir {
namespace cir {

// Forward declarations.
class LowerModule;

class CIRCXXABI {
  friend class LowerModule;

protected:
  LowerModule &LM;

  CIRCXXABI(LowerModule &LM) : LM(LM) {}

public:
  virtual ~CIRCXXABI();
};

/// Creates an Itanium-family ABI.
CIRCXXABI *CreateItaniumCXXABI(LowerModule &CGM);

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
