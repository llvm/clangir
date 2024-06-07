//===- ABIInfoImpl.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/ABIInfoImpl.h. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFOIMPL_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFOIMPL_H

#include "ABIInfo.h"
#include "CIRCXXABI.h"
#include "LowerFunctionInfo.h"

namespace mlir {
namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LowerFunctionInfo &FI,
                        const ABIInfo &Info);

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFOIMPL_H
