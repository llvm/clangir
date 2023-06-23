//===- CIRTypes.cpp - MLIR CIR Types --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// ClangIR holds back AST references when available.
#include "clang/AST/Decl.h"

static void printConstStructMembers(mlir::AsmPrinter &p, mlir::Type type,
                                    mlir::ArrayAttr members);
static mlir::ParseResult parseConstStructMembers(::mlir::AsmParser &parser,
                                                 mlir::Type &type,
                                                 mlir::ArrayAttr &members);

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

Attribute CIRDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &mnemonic, type, genAttr);
  if (parseResult.has_value())
    return genAttr;
  parser.emitError(typeLoc, "unknown attribute in CIR dialect");
  return Attribute();
}

void CIRDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  if (failed(generatedAttributePrinter(attr, os)))
    llvm_unreachable("unexpected CIR type kind");
}

static void printConstStructMembers(mlir::AsmPrinter &p, mlir::Type type,
                                    mlir::ArrayAttr members) {
  p << "{";
  unsigned i = 0, e = members.size();
  while (i < e) {
    p << members[i];
    if (e > 0 && i < e - 1)
      p << ",";
    i++;
  }
  p << "}";
}

static ParseResult parseConstStructMembers(::mlir::AsmParser &parser,
                                           mlir::Type &type,
                                           mlir::ArrayAttr &members) {
  SmallVector<mlir::Attribute, 4> elts;
  SmallVector<mlir::Type, 4> tys;
  if (parser
          .parseCommaSeparatedList(
              AsmParser::Delimiter::Braces,
              [&]() {
                Attribute attr;
                if (parser.parseAttribute(attr).succeeded()) {
                  elts.push_back(attr);
                  if (auto tyAttr = attr.dyn_cast<mlir::TypedAttr>()) {
                    tys.push_back(tyAttr.getType());
                    return success();
                  }
                  parser.emitError(parser.getCurrentLocation(),
                                   "expected a typed attribute");
                }
                return failure();
              })
          .failed())
    return failure();

  auto *ctx = parser.getContext();
  members = mlir::ArrayAttr::get(ctx, elts);
  type = mlir::cir::StructType::get(ctx, tys, "", /*body=*/true);
  return success();
}

LogicalResult ConstStructAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::Type type, ArrayAttr members) {
  auto sTy = type.dyn_cast_or_null<mlir::cir::StructType>();
  if (!sTy) {
    emitError() << "expected !cir.struct type";
    return failure();
  }

  if (sTy.getMembers().size() != members.size()) {
    emitError() << "number of elements must match";
    return failure();
  }

  unsigned attrIdx = 0;
  for (auto &member : sTy.getMembers()) {
    auto m = members[attrIdx].dyn_cast_or_null<TypedAttr>();
    if (!m) {
      emitError() << "expected mlir::TypedAttr attribute";
      return failure();
    }
    if (member != m.getType()) {
      emitError() << "element at index " << attrIdx << " has type "
                  << m.getType() << " but return type for this element is "
                  << member;
      return failure();
    }
    attrIdx++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LangInfoAttr definitions
//===----------------------------------------------------------------------===//

Attribute LangInfoAttr::parse(AsmParser &parser, Type odsType) {
  auto loc = parser.getCurrentLocation();
  if (parser.parseLess())
    return {};

  // Parse variable 'lang'.
  llvm::StringRef lang;
  if (parser.parseKeyword(&lang))
    return {};

  // Check if parsed value is a valid language.
  auto langEnum = symbolizeSourceLanguage(lang);
  if (!langEnum.has_value()) {
    parser.emitError(loc) << "invalid language keyword '" << lang << "'";
    return {};
  }

  if (parser.parseComma())
    return {};

  // Parse variable 'std'.
  llvm::StringRef std;
  if (parser.parseKeyword(&std))
    return {};

  // Check if parsed value is a valid standard.
  auto stdEnum = symbolizeLangStandard(std);
  if (!stdEnum.has_value()) {
    parser.emitError(loc) << "invalid language standard '" << std << "'";
    return {};
  }

  if (parser.parseGreater())
    return {};

  // Create and validate lang info attribute.
  auto attr = get(parser.getContext(), langEnum.value(), stdEnum.value());
  if ((attr.isC() && !attr.isCStd()) || (attr.isCXX() && !attr.isCXXStd())) {
    parser.emitError(loc) << "invalid " << lang << " standard '" << std << "'";
    return {};
  }

  return attr;
}

void LangInfoAttr::print(AsmPrinter &printer) const {
  printer << "<" << getLang() << ", " << getStd() << '>';
}

bool LangInfoAttr::isCStd() const {
  auto std = getStd();
  using LS = LangStandard;
  return std == LS::C89 || std == LS::C94 || std == LS::C99 || std == LS::C11 ||
         std == LS::C17 || std == LS::C2X;
}

bool LangInfoAttr::isCXXStd() const {
  auto std = getStd();
  using LS = LangStandard;
  return std == LS::CXX98 || std == LS::CXX11 || std == LS::CXX14 ||
         std == LS::CXX17 || std == LS::CXX20 || std == LS::CXX23 ||
         std == LS::CXX26;
}

//===----------------------------------------------------------------------===//
// IntAttr definitions
//===----------------------------------------------------------------------===//

Attribute IntAttr::parse(AsmParser &parser, Type odsType) {
  mlir::APInt APValue;

  if (!odsType.isa<IntType>())
    return {};
  auto type = odsType.cast<IntType>();

  // Consume the '<' symbol.
  if (parser.parseLess())
    return {};

  // Fetch arbitrary precision integer value.
  if (type.isSigned()) {
    int64_t value;
    if (parser.parseInteger(value))
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
    if (APValue.getSExtValue() != value)
      parser.emitError(parser.getCurrentLocation(),
                       "integer value too large for the given type");
  } else {
    uint64_t value;
    if (parser.parseInteger(value))
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
    APValue = mlir::APInt(type.getWidth(), value, type.isSigned());
    if (APValue.getZExtValue() != value)
      parser.emitError(parser.getCurrentLocation(),
                       "integer value too large for the given type");
  }

  // Consume the '>' symbol.
  if (parser.parseGreater())
    return {};

  return IntAttr::get(type, APValue);
}

void IntAttr::print(AsmPrinter &printer) const {
  auto type = getType().cast<IntType>();
  printer << '<';
  if (type.isSigned())
    printer << getSInt();
  else
    printer << getUInt();
  printer << '>';
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type type, APInt value) {
  if (!type.isa<IntType>()) {
    emitError() << "expected 'simple.int' type";
    return failure();
  }

  auto intType = type.cast<IntType>();
  if (value.getBitWidth() != intType.getWidth()) {
    emitError() << "type and value bitwidth mismatch: " << intType.getWidth()
                << " != " << value.getBitWidth();
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//

void CIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"
      >();
}
