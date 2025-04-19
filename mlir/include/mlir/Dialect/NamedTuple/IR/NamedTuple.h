//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NamedTuple dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NAMED_TUPLE_IR_NAMED_TUPLE_H
#define MLIR_DIALECT_NAMED_TUPLE_IR_NAMED_TUPLE_H

#include "mlir/Dialect/NamedTuple/IR/NamedTupleDialect.h"
#include "mlir/Dialect/NamedTuple/IR/NamedTupleTypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/NamedTuple/IR/NamedTuple.h.inc"

#endif // MLIR_DIALECT_NAMED_TUPLE_IR_NAMED_TUPLE_H
