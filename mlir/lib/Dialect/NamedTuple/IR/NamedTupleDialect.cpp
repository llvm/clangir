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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NamedTuple/IR/NamedTuple.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include "mlir/Dialect/NamedTuple/IR/NamedTupleDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/NamedTuple/IR/NamedTupleTypes.cpp.inc"

// Dummy implementation for now
// TODO(cir)
llvm::TypeSize mlir::named_tuple::NamedTupleType::getTypeSizeInBits(
    mlir::DataLayout const &,
    llvm::ArrayRef<mlir::DataLayoutEntryInterface>) const {
  llvm_unreachable("getTypeSizeInBits() not implemented");
  return llvm::TypeSize::getFixed(8);
}

uint64_t mlir::named_tuple::NamedTupleType::getABIAlignment(
    mlir::DataLayout const &,
    llvm::ArrayRef<mlir::DataLayoutEntryInterface>) const {
  llvm_unreachable("getABIAlignment() not implemented");
  return 8;
}

uint64_t mlir::named_tuple::NamedTupleType::getPreferredAlignment(
    mlir::DataLayout const &,
    llvm::ArrayRef<mlir::DataLayoutEntryInterface>) const {
  llvm_unreachable("getPreferredAlignment() not implemented");
  return 8;
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "mlir/Dialect/NamedTuple/IR/NamedTuple.cpp.inc"

bool mlir::named_tuple::CastOp::areCastCompatible(mlir::TypeRange inputs,
                                                  mlir::TypeRange outputs) {
  return true;
}

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
