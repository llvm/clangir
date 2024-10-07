//===- CIRLowerContext.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/ASTContext.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRLowerContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>
#include <utility>

namespace mlir {
namespace cir {

CIRLowerContext::CIRLowerContext(ModuleOp module, clang::LangOptions lOpts)
    : MLIRCtx(module.getContext()), LangOpts(std::move(lOpts)) {}

CIRLowerContext::~CIRLowerContext() = default;

clang::TypeInfo CIRLowerContext::getTypeInfo(Type t) const {
  // TODO(cir): Memoize type info.

  clang::TypeInfo ti = getTypeInfoImpl(t);
  return ti;
}

/// getTypeInfoImpl - Return the size of the specified type, in bits.  This
/// method does not work on incomplete types.
///
/// FIXME: Pointers into different addr spaces could have different sizes and
/// alignment requirements: getPointerInfo should take an AddrSpace, this
/// should take a QualType, &c.
clang::TypeInfo CIRLowerContext::getTypeInfoImpl(const Type t) const {
  uint64_t width = 0;
  unsigned align = 8;
  clang::AlignRequirementKind alignRequirement =
      clang::AlignRequirementKind::None;

  // TODO(cir): We should implement a better way to identify type kinds and use
  // builting data layout interface for this.
  auto typeKind = clang::Type::Builtin;
  if (isa<IntType, SingleType, DoubleType, BoolType>(t)) {
    typeKind = clang::Type::Builtin;
  } else if (isa<StructType>(t)) {
    typeKind = clang::Type::Record;
  } else {
    llvm_unreachable("Unhandled type class");
  }

  // FIXME(cir): Here we fetch the width and alignment of a type considering the
  // current target. We can likely improve this using MLIR's data layout, or
  // some other interface, to abstract this away (e.g. type.getWidth() &
  // type.getAlign()). Verify if data layout suffices because this would involve
  // some other types such as vectors and complex numbers.
  // FIXME(cir): In the original codegen, this receives an AST type, meaning it
  // differs chars from integers, something that is not possible with the
  // current level of CIR.
  switch (typeKind) {
  case clang::Type::Builtin: {
    if (auto intTy = dyn_cast<IntType>(t)) {
      // NOTE(cir): This assumes int types are already ABI-specific.
      // FIXME(cir): Use data layout interface here instead.
      width = intTy.getWidth();
      // FIXME(cir): Use the proper getABIAlignment method here.
      align = std::ceil((float)width / 8) * 8;
      break;
    }
    if (auto boolTy = dyn_cast<BoolType>(t)) {
      width = Target->getFloatWidth();
      align = Target->getFloatAlign();
      break;
    }
    if (auto floatTy = dyn_cast<SingleType>(t)) {
      width = Target->getFloatWidth();
      align = Target->getFloatAlign();
      break;
    }
    if (auto doubleTy = dyn_cast<DoubleType>(t)) {
      width = Target->getDoubleWidth();
      align = Target->getDoubleAlign();
      break;
    }
    llvm_unreachable("Unknown builtin type!");
    break;
  }
  case clang::Type::Record: {
    const auto rt = dyn_cast<StructType>(t);
    assert(!::cir::MissingFeatures::tagTypeClassAbstraction());

    // Only handle TagTypes (names types) for now.
    assert(rt.getName() && "Anonymous record is NYI");

    // NOTE(cir): Clang does some hanlding of invalid tagged declarations here.
    // Not sure if this is necessary in CIR.

    if (::cir::MissingFeatures::typeGetAsEnumType()) {
      llvm_unreachable("NYI");
    }

    const CIRRecordLayout &layout = getCIRRecordLayout(rt);
    width = toBits(layout.getSize());
    align = toBits(layout.getAlignment());
    assert(!::cir::MissingFeatures::recordDeclHasAlignmentAttr());
    break;
  }
  default:
    llvm_unreachable("Unhandled type class");
  }

  assert(llvm::isPowerOf2_32(align) && "Alignment must be power of 2");
  return clang::TypeInfo(width, align, alignRequirement);
}

Type CIRLowerContext::initBuiltinType(clang::BuiltinType::Kind k) {
  Type ty;

  // NOTE(cir): Clang does more stuff here. Not sure if we need to do the same.
  assert(!::cir::MissingFeatures::qualifiedTypes());
  switch (k) {
  case clang::BuiltinType::Char_S:
    ty = IntType::get(getMLIRContext(), 8, true);
    break;
  default:
    llvm_unreachable("NYI");
  }

  Types.push_back(ty);
  return ty;
}

void CIRLowerContext::initBuiltinTypes(const clang::TargetInfo &target,
                                       const clang::TargetInfo *auxTarget) {
  assert((!this->Target || this->Target == &target) &&
         "Incorrect target reinitialization");
  this->Target = &target;
  this->AuxTarget = auxTarget;

  // C99 6.2.5p3.
  if (LangOpts.CharIsSigned)
    CharTy = initBuiltinType(clang::BuiltinType::Char_S);
  else
    llvm_unreachable("NYI");
}

/// Convert a size in bits to a size in characters.
clang::CharUnits CIRLowerContext::toCharUnitsFromBits(int64_t bitSize) const {
  return clang::CharUnits::fromQuantity(bitSize / getCharWidth());
}

/// Convert a size in characters to a size in characters.
int64_t CIRLowerContext::toBits(clang::CharUnits charSize) const {
  return charSize.getQuantity() * getCharWidth();
}

clang::TypeInfoChars CIRLowerContext::getTypeInfoInChars(Type t) const {
  if (auto arrTy = dyn_cast<ArrayType>(t))
    llvm_unreachable("NYI");
  clang::TypeInfo info = getTypeInfo(t);
  return clang::TypeInfoChars(toCharUnitsFromBits(info.Width),
                              toCharUnitsFromBits(info.Align),
                              info.AlignRequirement);
}

bool CIRLowerContext::isPromotableIntegerType(Type t) const {
  // HLSL doesn't promote all small integer types to int, it
  // just uses the rank-based promotion rules for all types.
  if (::cir::MissingFeatures::langOpts())
    llvm_unreachable("NYI");

  // FIXME(cir): CIR does not distinguish between char, short, etc. So we just
  // assume it is promotable if smaller than 32 bits. This is wrong since, for
  // example, Char32 is promotable. Improve CIR or add an AST query here.
  if (auto intTy = dyn_cast<IntType>(t)) {
    return cast<IntType>(t).getWidth() < 32;
  }

  // Bool are also handled here for codegen parity.
  if (auto boolTy = dyn_cast<BoolType>(t)) {
    return true;
  }

  // Enumerated types are promotable to their compatible integer types
  // (C99 6.3.1.1) a.k.a. its underlying type (C++ [conv.prom]p2).
  // TODO(cir): CIR doesn't know if a integer originated from an enum. Improve
  // CIR or add an AST query here.
  if (::cir::MissingFeatures::typeGetAsEnumType()) {
    llvm_unreachable("NYI");
  }

  return false;
}

} // namespace cir
} // namespace mlir
