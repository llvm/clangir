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
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// ClangIR holds back AST references when available.
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"

static void printStructMembers(mlir::AsmPrinter &p, mlir::ArrayAttr members);
static mlir::ParseResult parseStructMembers(::mlir::AsmParser &parser,
                                            mlir::ArrayAttr &members);

static void printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value,
                              mlir::Type ty);
static mlir::ParseResult
parseFloatLiteral(mlir::AsmParser &parser,
                  mlir::FailureOr<llvm::APFloat> &value, mlir::Type ty);

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// CIR AST Attr helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cir {

mlir::Attribute makeFuncDeclAttr(const clang::Decl *decl,
                                 mlir::MLIRContext *ctx) {
  return llvm::TypeSwitch<const clang::Decl *, mlir::Attribute>(decl)
      .Case([ctx](const clang::CXXConstructorDecl *ast) {
        return ASTCXXConstructorDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::CXXConversionDecl *ast) {
        return ASTCXXConversionDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::CXXDestructorDecl *ast) {
        return ASTCXXDestructorDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::CXXMethodDecl *ast) {
        return ASTCXXMethodDeclAttr::get(ctx, ast);
      })
      .Case([ctx](const clang::FunctionDecl *ast) {
        return ASTFunctionDeclAttr::get(ctx, ast);
      })
      .Default([](auto) {
        llvm_unreachable("unexpected Decl kind");
        return mlir::Attribute();
      });
}

} // namespace cir
} // namespace mlir

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

static void printStructMembers(mlir::AsmPrinter &printer,
                               mlir::ArrayAttr members) {
  printer << '{';
  llvm::interleaveComma(members, printer);
  printer << '}';
}

static ParseResult parseStructMembers(mlir::AsmParser &parser,
                                      mlir::ArrayAttr &members) {
  SmallVector<mlir::Attribute, 4> elts;

  auto delimiter = AsmParser::Delimiter::Braces;
  auto result = parser.parseCommaSeparatedList(delimiter, [&]() {
    mlir::TypedAttr attr;
    if (parser.parseAttribute(attr).failed())
      return mlir::failure();
    elts.push_back(attr);
    return mlir::success();
  });

  if (result.failed())
    return mlir::failure();

  members = mlir::ArrayAttr::get(parser.getContext(), elts);
  return mlir::success();
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
// LangAttr definitions
//===----------------------------------------------------------------------===//

Attribute LangAttr::parse(AsmParser &parser, Type odsType) {
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

  if (parser.parseGreater())
    return {};

  return get(parser.getContext(), langEnum.value());
}

void LangAttr::print(AsmPrinter &printer) const {
  printer << "<" << getLang() << '>';
}

//===----------------------------------------------------------------------===//
// ConstPtrAttr definitions
//===----------------------------------------------------------------------===//

Attribute ConstPtrAttr::parse(AsmParser &parser, Type odsType) {
  uint64_t value;

  if (!odsType.isa<cir::PointerType>())
    return {};

  // Consume the '<' symbol.
  if (parser.parseLess())
    return {};

  if (parser.parseOptionalKeyword("null").succeeded()) {
    value = 0;
  } else {
    if (parser.parseInteger(value))
      parser.emitError(parser.getCurrentLocation(), "expected integer value");
  }

  // Consume the '>' symbol.
  if (parser.parseGreater())
    return {};

  return ConstPtrAttr::get(odsType, value);
}

void ConstPtrAttr::print(AsmPrinter &printer) const {
  printer << '<';
  if (isNullValue())
    printer << "null";
  else
    printer << getValue();
  printer << '>';
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
// FPAttr definitions
//===----------------------------------------------------------------------===//

static void printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value,
                              mlir::Type ty) {
  p << value;
}

static mlir::ParseResult
parseFloatLiteral(mlir::AsmParser &parser,
                  mlir::FailureOr<llvm::APFloat> &value, mlir::Type ty) {
  double rawValue;
  if (parser.parseFloat(rawValue)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected floating-point value");
  }

  auto losesInfo = false;
  value.emplace(rawValue);

  auto tyFpInterface = ty.dyn_cast<cir::CIRFPTypeInterface>();
  if (!tyFpInterface) {
    // Parsing of the current floating-point literal has succeeded, but the
    // given attribute type is invalid. This error will be reported later when
    // the attribute is being verified.
    return success();
  }

  value->convert(tyFpInterface.getFloatSemantics(),
                 llvm::RoundingMode::TowardZero, &losesInfo);
  return success();
}

cir::FPAttr cir::FPAttr::getZero(mlir::Type type) {
  return get(type,
             APFloat::getZero(
                 type.cast<cir::CIRFPTypeInterface>().getFloatSemantics()));
}

LogicalResult cir::FPAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type type, APFloat value) {
  auto fltTypeInterface = type.dyn_cast<cir::CIRFPTypeInterface>();
  if (!fltTypeInterface) {
    emitError() << "expected floating-point type";
    return failure();
  }
  if (APFloat::SemanticsToEnum(fltTypeInterface.getFloatSemantics()) !=
      APFloat::SemanticsToEnum(value.getSemantics())) {
    emitError() << "floating-point semantics mismatch";
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
