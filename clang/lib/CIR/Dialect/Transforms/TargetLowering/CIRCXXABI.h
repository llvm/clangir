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

#include "LowerFunctionInfo.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Target/AArch64.h"

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

  /// If the C++ ABI requires the given type be returned in a particular way,
  /// this method sets RetAI and returns true.
  virtual bool classifyReturnType(LowerFunctionInfo &FI) const = 0;

  /// Specify how one should pass an argument of a record type.
  enum RecordArgABI {
    /// Pass it using the normal C aggregate rules for the ABI, potentially
    /// introducing extra copies and passing some or all of it in registers.
    RAA_Default = 0,

    /// Pass it on the stack using its defined layout.  The argument must be
    /// evaluated directly into the correct stack position in the arguments
    /// area,
    /// and the call machinery must not move it or introduce extra copies.
    RAA_DirectInMemory,

    /// Pass it as a pointer to temporary memory.
    RAA_Indirect
  };

  /// Returns how an argument of the given record type should be passed.
  /// FIXME(cir): This expects a CXXRecordDecl! Not any record type.
  virtual RecordArgABI getRecordArgABI(const StructType RD) const = 0;
};

/// Creates an Itanium-family ABI.
CIRCXXABI *CreateItaniumCXXABI(LowerModule &CGM);

} // namespace cir
} // namespace mlir

// FIXME(cir): Merge this into the CIRCXXABI class above. To do so, this code
// should be updated to follow some level of codegen parity.
namespace cir {

class LoweringPrepareCXXABI {
public:
  static LoweringPrepareCXXABI *createItaniumABI();
  static LoweringPrepareCXXABI *createAArch64ABI(::cir::AArch64ABIKind k);

  virtual mlir::Value lowerVAArg(CIRBaseBuilderTy &builder,
                                 mlir::cir::VAArgOp op,
                                 const cir::CIRDataLayout &datalayout) = 0;
  virtual ~LoweringPrepareCXXABI() {}

  virtual mlir::Value lowerDynamicCast(CIRBaseBuilderTy &builder,
                                       clang::ASTContext &astCtx,
                                       mlir::cir::DynamicCastOp op) = 0;
};
} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRCXXABI_H
