//===- CIRAttrs.cpp - MLIR CIR Attributes ---------------------------------===//
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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "clang/CIR/Interfaces/CIRTypeInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

// ClangIR holds back AST references when available.
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"

//===-----------------------------------------------------------------===//
// RecordMembers
//===-----------------------------------------------------------------===//

static void printRecordMembers(mlir::AsmPrinter &p, mlir::ArrayAttr members);
static mlir::ParseResult parseRecordMembers(::mlir::AsmParser &parser,
                                            mlir::ArrayAttr &members);

//===-----------------------------------------------------------------===//
// IntLiteral
//===-----------------------------------------------------------------===//

static void printIntLiteral(mlir::AsmPrinter &p, llvm::APInt value,
                            cir::IntTypeInterface ty);

static mlir::ParseResult parseIntLiteral(mlir::AsmParser &parser,
                                         llvm::APInt &value,
                                         cir::IntTypeInterface ty);

//===-----------------------------------------------------------------===//
// FloatLiteral
//===-----------------------------------------------------------------===//

static void printFloatLiteral(mlir::AsmPrinter &p, llvm::APFloat value,
                              mlir::Type ty);
static mlir::ParseResult
parseFloatLiteral(mlir::AsmParser &parser,
                  mlir::FailureOr<llvm::APFloat> &value,
                  cir::FPTypeInterface fpType);

static mlir::ParseResult parseConstPtr(mlir::AsmParser &parser,
                                       mlir::IntegerAttr &value);

static void printConstPtr(mlir::AsmPrinter &p, mlir::IntegerAttr value);

#define GET_ATTRDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsAttributes.cpp.inc"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// CIR AST Attr helpers
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// General CIR parsing / printing
//===----------------------------------------------------------------------===//

Attribute CIRDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  llvm::StringRef mnemonic;
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

static void printRecordMembers(mlir::AsmPrinter &printer,
                               mlir::ArrayAttr members) {
  printer << '{';
  llvm::interleaveComma(members, printer);
  printer << '}';
}

static ParseResult parseRecordMembers(mlir::AsmParser &parser,
                                      mlir::ArrayAttr &members) {
  llvm::SmallVector<mlir::Attribute, 4> elts;

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

LogicalResult
ConstRecordAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        mlir::Type type, ArrayAttr members) {
  auto sTy = mlir::dyn_cast_if_present<cir::RecordType>(type);
  if (!sTy)
    return emitError() << "expected !cir.record type";

  if (sTy.getMembers().size() != members.size())
    return emitError() << "number of elements must match";

  unsigned attrIdx = 0;
  for (auto &member : sTy.getMembers()) {
    auto m = dyn_cast_if_present<TypedAttr>(members[attrIdx]);
    if (!m)
      return emitError() << "expected mlir::TypedAttr attribute";
    if (member != m.getType())
      return emitError() << "element at index " << attrIdx << " has type "
                         << m.getType()
                         << " but return type for this element is " << member;
    attrIdx++;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OptInfoAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult OptInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  unsigned level, unsigned size) {
  if (level > 3)
    return emitError()
           << "optimization level must be between 0 and 3 inclusive";
  if (size > 2)
    return emitError()
           << "size optimization level must be between 0 and 2 inclusive";
  return success();
}

//===----------------------------------------------------------------------===//
// ConstPtrAttr definitions
//===----------------------------------------------------------------------===//

// TODO: Consider encoding the null value differently and use conditional
// assembly format instead of custom parsing/printing.
static ParseResult parseConstPtr(AsmParser &parser, mlir::IntegerAttr &value) {

  if (parser.parseOptionalKeyword("null").succeeded()) {
    value = parser.getBuilder().getI64IntegerAttr(0);
    return success();
  }

  return parser.parseAttribute(value);
}

static void printConstPtr(AsmPrinter &p, mlir::IntegerAttr value) {
  if (!value.getInt())
    p << "null";
  else
    p << value;
}

//===----------------------------------------------------------------------===//
// IntAttr definitions
//===----------------------------------------------------------------------===//

template <typename IntT>
static bool isTooLargeForType(const mlir::APInt &v, IntT expectedValue) {
  if constexpr (std::is_signed_v<IntT>) {
    return v.getSExtValue() != expectedValue;
  } else {
    return v.getZExtValue() != expectedValue;
  }
}

template <typename IntT>
static ParseResult parseIntLiteralImpl(mlir::AsmParser &p, llvm::APInt &value,
                                       cir::IntTypeInterface ty) {
  IntT ivalue;
  const bool isSigned = ty.isSigned();
  if (p.parseInteger(ivalue))
    return p.emitError(p.getCurrentLocation(), "expected integer value");

  value = mlir::APInt(ty.getWidth(), ivalue, isSigned, /*implicitTrunc=*/true);
  if (isTooLargeForType(value, ivalue))

    return p.emitError(p.getCurrentLocation(),
                       "integer value too large for the given type");

  return success();
}

mlir::ParseResult parseIntLiteral(mlir::AsmParser &parser, llvm::APInt &value,
                                  cir::IntTypeInterface ty) {
  if (ty.isSigned())
    return parseIntLiteralImpl<int64_t>(parser, value, ty);
  return parseIntLiteralImpl<uint64_t>(parser, value, ty);
}

void printIntLiteral(mlir::AsmPrinter &p, llvm::APInt value,
                     cir::IntTypeInterface ty) {
  if (ty.isSigned())
    p << value.getSExtValue();
  else
    p << value.getZExtValue();
}

LogicalResult IntAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              cir::IntTypeInterface type, llvm::APInt value) {
  if (value.getBitWidth() != type.getWidth())
    return emitError() << "type and value bitwidth mismatch: "
                       << type.getWidth() << " != " << value.getBitWidth();
  return success();
}

//===----------------------------------------------------------------------===//
// FPAttr definitions
//===----------------------------------------------------------------------===//

static void printFloatLiteral(AsmPrinter &p, APFloat value, Type ty) {
  p << value;
}

static ParseResult parseFloatLiteral(AsmParser &parser,
                                     FailureOr<APFloat> &value,
                                     cir::FPTypeInterface fpType) {

  APFloat parsedValue(0.0);
  if (parser.parseFloat(fpType.getFloatSemantics(), parsedValue))
    return failure();

  value.emplace(parsedValue);
  return success();
}

FPAttr FPAttr::getZero(Type type) {
  return get(type,
             APFloat::getZero(
                 mlir::cast<cir::FPTypeInterface>(type).getFloatSemantics()));
}

LogicalResult FPAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                             cir::FPTypeInterface fpType, APFloat value) {
  if (APFloat::SemanticsToEnum(fpType.getFloatSemantics()) !=
      APFloat::SemanticsToEnum(value.getSemantics()))
    return emitError() << "floating-point semantics mismatch";

  return success();
}

//===----------------------------------------------------------------------===//
// ComplexAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult ComplexAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                  cir::ComplexType type, mlir::TypedAttr real,
                                  mlir::TypedAttr imag) {
  auto elemType = type.getElementType();
  if (real.getType() != elemType)
    return emitError()
           << "type of the real part does not match the complex type";

  if (imag.getType() != elemType)
    return emitError()
           << "type of the imaginary part does not match the complex type";

  return success();
}

//===----------------------------------------------------------------------===//
// CmpThreeWayInfoAttr definitions
//===----------------------------------------------------------------------===//

std::string CmpThreeWayInfoAttr::getAlias() const {
  std::string alias = "cmp3way_info";

  if (getOrdering() == CmpOrdering::Strong)
    alias.append("_strong_");
  else
    alias.append("_partial_");

  auto appendInt = [&](int64_t value) {
    if (value < 0) {
      alias.push_back('n');
      value = -value;
    }
    alias.append(std::to_string(value));
  };

  alias.append("lt");
  appendInt(getLt());
  alias.append("eq");
  appendInt(getEq());
  alias.append("gt");
  appendInt(getGt());

  if (auto unordered = getUnordered()) {
    alias.append("un");
    appendInt(unordered.value());
  }

  return alias;
}

LogicalResult
CmpThreeWayInfoAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            CmpOrdering ordering, int64_t lt, int64_t eq,
                            int64_t gt, std::optional<int64_t> unordered) {
  // The presense of unordered must match the value of ordering.
  if (ordering == CmpOrdering::Strong && unordered)
    return emitError() << "strong ordering does not include unordered ordering";

  if (ordering == CmpOrdering::Partial && !unordered)
    return emitError() << "partial ordering lacks unordered ordering";

  return success();
}

//===----------------------------------------------------------------------===//
// DataMemberAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
DataMemberAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       cir::DataMemberType ty,
                       std::optional<unsigned> memberIndex) {
  if (!memberIndex.has_value()) {
    // DataMemberAttr without a given index represents a null value.
    return success();
  }

  auto clsRecordTy = ty.getClsTy();
  if (clsRecordTy.isIncomplete())
    return emitError()
           << "incomplete 'cir.record' cannot be used to build a non-null "
              "data member pointer";

  auto memberIndexValue = memberIndex.value();
  if (memberIndexValue >= clsRecordTy.getNumElements())
    return emitError()
           << "member index of a #cir.data_member attribute is out of range";

  auto memberTy = clsRecordTy.getMembers()[memberIndexValue];
  if (memberTy != ty.getMemberTy())
    return emitError()
           << "member type of a #cir.data_member attribute must match the "
              "attribute type";

  return success();
}

//===----------------------------------------------------------------------===//
// MethodAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult MethodAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 cir::MethodType type,
                                 std::optional<FlatSymbolRefAttr> symbol,
                                 std::optional<uint64_t> vtable_offset) {
  if (symbol.has_value() && vtable_offset.has_value())
    return emitError()
           << "at most one of symbol and vtable_offset can be present "
              "in #cir.method";

  return success();
}

Attribute MethodAttr::parse(AsmParser &parser, Type odsType) {
  auto ty = mlir::cast<cir::MethodType>(odsType);

  if (parser.parseLess())
    return {};

  // Try to parse the null pointer constant.
  if (parser.parseOptionalKeyword("null").succeeded()) {
    if (parser.parseGreater())
      return {};
    return get(ty);
  }

  // Try to parse a flat symbol ref for a pointer to non-virtual member
  // function.
  FlatSymbolRefAttr symbol;
  auto parseSymbolRefResult = parser.parseOptionalAttribute(symbol);
  if (parseSymbolRefResult.has_value()) {
    if (parseSymbolRefResult.value().failed())
      return {};
    if (parser.parseGreater())
      return {};
    return get(ty, symbol);
  }

  // Parse a uint64 that represents the vtable offset.
  std::uint64_t vtableOffset = 0;
  if (parser.parseKeyword("vtable_offset"))
    return {};
  if (parser.parseEqual())
    return {};
  if (parser.parseInteger(vtableOffset))
    return {};

  if (parser.parseGreater())
    return {};

  return get(ty, vtableOffset);
}

void MethodAttr::print(AsmPrinter &printer) const {
  auto symbol = getSymbol();
  auto vtableOffset = getVtableOffset();

  printer << '<';
  if (symbol.has_value()) {
    printer << *symbol;
  } else if (vtableOffset.has_value()) {
    printer << "vtable_offset = " << *vtableOffset;
  } else {
    printer << "null";
  }
  printer << '>';
}

//===----------------------------------------------------------------------===//
// GlobalAnnotationValuesAttr definitions
//===----------------------------------------------------------------------===//

LogicalResult
GlobalAnnotationValuesAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   mlir::ArrayAttr annotations) {
  if (annotations.empty())
    return emitError() << "GlobalAnnotationValuesAttr should at least have "
                          "one annotation";

  for (auto &entry : annotations) {
    auto annoEntry = ::mlir::dyn_cast<mlir::ArrayAttr>(entry);
    if (!annoEntry)
      return emitError()
             << "Element of GlobalAnnotationValuesAttr annotations array"
                " must be an array";

    if (annoEntry.size() != 2)
      return emitError()
             << "Element of GlobalAnnotationValuesAttr annotations array"
             << " must be a 2-element array and you have " << annoEntry.size();

    if (!mlir::isa<mlir::StringAttr>(annoEntry[0]))
      return emitError()
             << "Element of GlobalAnnotationValuesAttr annotations"
                "array must start with a string, which is the name of "
                "global op or func it annotates";

    if (!mlir::isa<cir::AnnotationAttr>(annoEntry[1]))
      return emitError() << "The second element of GlobalAnnotationValuesAttr"
                            "annotations array element must be of "
                            "type AnnotationValueAttr";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicCastInfoAtttr definitions
//===----------------------------------------------------------------------===//

std::string DynamicCastInfoAttr::getAlias() const {
  // The alias looks like: `dyn_cast_info_<src>_<dest>`

  std::string alias = "dyn_cast_info_";

  alias.append(getSrcRtti().getSymbol().getValue());
  alias.push_back('_');
  alias.append(getDestRtti().getSymbol().getValue());

  return alias;
}

LogicalResult DynamicCastInfoAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, cir::GlobalViewAttr srcRtti,
    cir::GlobalViewAttr destRtti, mlir::FlatSymbolRefAttr runtimeFunc,
    mlir::FlatSymbolRefAttr badCastFunc, cir::IntAttr offsetHint) {
  auto isRttiPtr = [](mlir::Type ty) {
    // RTTI pointers are !cir.ptr<!u8i>.

    auto ptrTy = mlir::dyn_cast<cir::PointerType>(ty);
    if (!ptrTy)
      return false;

    auto pointeeIntTy = mlir::dyn_cast<cir::IntType>(ptrTy.getPointee());
    if (!pointeeIntTy)
      return false;

    return pointeeIntTy.isUnsigned() && pointeeIntTy.getWidth() == 8;
  };

  if (!isRttiPtr(srcRtti.getType()))
    return emitError() << "srcRtti must be an RTTI pointer";

  if (!isRttiPtr(destRtti.getType()))
    return emitError() << "destRtti must be an RTTI pointer";

  return success();
}

//===----------------------------------------------------------------------===//
// AddressSpaceAttr definitions
//===----------------------------------------------------------------------===//

std::optional<int32_t>
AddressSpaceAttr::getValueFromLangAS(clang::LangAS langAS) {
  using clang::LangAS;
  switch (langAS) {
  case LangAS::Default:
    // Default address space should be encoded as a null attribute.
    return std::nullopt;
  case LangAS::opencl_global:
    return Kind::offload_global;
  case LangAS::opencl_local:
  case LangAS::cuda_shared:
    // Local means local among the work-group (OpenCL) or block (CUDA).
    // All threads inside the kernel can access local memory.
    return Kind::offload_local;
  case LangAS::cuda_device:
    return Kind::offload_global;
  case LangAS::opencl_constant:
  case LangAS::cuda_constant:
    return Kind::offload_constant;
  case LangAS::opencl_private:
    return Kind::offload_private;
  case LangAS::opencl_generic:
    return Kind::offload_generic;
  case LangAS::opencl_global_device:
  case LangAS::opencl_global_host:
  case LangAS::sycl_global:
  case LangAS::sycl_global_device:
  case LangAS::sycl_global_host:
  case LangAS::sycl_local:
  case LangAS::sycl_private:
  case LangAS::ptr32_sptr:
  case LangAS::ptr32_uptr:
  case LangAS::ptr64:
  case LangAS::hlsl_groupshared:
  case LangAS::wasm_funcref:
    llvm_unreachable("NYI");
  default:
    // Target address space offset arithmetics
    return clang::toTargetAddressSpace(langAS) + kFirstTargetASValue;
  }
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
