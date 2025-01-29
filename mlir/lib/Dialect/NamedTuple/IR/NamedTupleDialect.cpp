//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Concrete implementation of NamedTuple dialect
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/NamedTuple/IR/NamedTuple.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "mlir/Dialect/NamedTuple/IR/NamedTupleDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/NamedTuple/IR/NamedTupleTypes.cpp.inc"

void mlir::named_tuple::NamedTupleDialect::initialize() {
  // Add the defined types to the dialect.
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/NamedTuple/IR/NamedTupleTypes.cpp.inc"
      >();

  // Add the defined operations to the dialect.
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/NamedTuple/IR/NamedTuple.cpp.inc"
      >();
}
