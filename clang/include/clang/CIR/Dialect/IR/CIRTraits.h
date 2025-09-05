//===- CIRTraits.h - MLIR CIR Traits ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the traits in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRTRAITS_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRTRAITS_H

#include "mlir/Dialect/Traits.h"

namespace mlir {
namespace OpTrait {
namespace cir {

/// This class provides the API for ops that are known to be terminators.
template <typename ConcreteType>
class Breakable : public TraitBase<ConcreteType, Breakable> {};

} // namespace cir
} // namespace OpTrait
} // namespace mlir

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRTRAITS_H
