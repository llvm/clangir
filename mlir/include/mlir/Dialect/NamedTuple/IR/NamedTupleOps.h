//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operations in the NamedTuple dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_NAMED_TUPLE_IR_NAMED_TUPLE_OPS_H
#define MLIR_DIALECT_NAMED_TUPLE_IR_NAMED_TUPLE_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/NamedTuple/IR/NamedTuple.h"
#include "mlir/Dialect/NamedTuple/IR/NamedTupleTypes.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/NamedTuple/IR/NamedTupleOps.h.inc"

#endif // MLIR_DIALECT_NAMED_TUPLE_IR_NAMED_TUPLE_OPS_H
