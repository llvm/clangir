//===- CIRAttrs.h - MLIR CIR Attrs ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRATTRS_H
#define CLANG_CIR_DIALECT_IR_CIRATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIRTypeInterfaces.h"

namespace cir {
inline constexpr uint32_t DefaultGlobalCtorDtorPriority = 65535;
} // namespace cir

//===----------------------------------------------------------------------===//
// CIR Dialect Attrs
//===----------------------------------------------------------------------===//

namespace clang {
class FunctionDecl;
class RecordDecl;
class VarDecl;
} // namespace clang

namespace cir {
class ArrayType;
class BoolType;
class ComplexType;
class DataMemberType;
class IntType;
class MethodType;
class PointerType;
class RecordType;
class VectorType;
} // namespace cir

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.h.inc"

template <> struct ::mlir::FieldParser<cir::TBAAAttr, cir::TBAAAttr> {
  static mlir::FailureOr<cir::TBAAAttr> parse(mlir::AsmParser &parser) {
    mlir::Attribute attribute;
    if (parser.parseAttribute(attribute))
      return mlir::failure();
    if (auto omnipotentChar =
            mlir::dyn_cast<cir::TBAAOmnipotentCharAttr>(attribute))
      return omnipotentChar;
    if (auto vtablePtr = mlir::dyn_cast<cir::TBAAVTablePointerAttr>(attribute))
      return vtablePtr;
    if (auto scalar = mlir::dyn_cast<cir::TBAAScalarAttr>(attribute))
      return scalar;
    if (auto tag = mlir::dyn_cast<cir::TBAATagAttr>(attribute))
      return tag;
    if (auto structAttr = mlir::dyn_cast<cir::TBAAStructAttr>(attribute))
      return structAttr;
    return parser.emitError(parser.getCurrentLocation(), "Expected TBAAAttr");
  }
};

#endif // CLANG_CIR_DIALECT_IR_CIRATTRS_H
