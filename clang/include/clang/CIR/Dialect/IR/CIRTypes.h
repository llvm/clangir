//===- CIRTypes.h - MLIR CIR Types ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_IR_CIRTYPES_H_
#define MLIR_DIALECT_CIR_IR_CIRTYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Interfaces/CIRFPTypeInterface.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"

//===----------------------------------------------------------------------===//
// CIR StructType
//
// The base type for all RecordDecls.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cir {

namespace detail {
struct StructTypeStorage;
} // namespace detail

/// Each unique clang::RecordDecl is mapped to a `cir.struct` and any object in
/// C/C++ that has a struct type will have a `cir.struct` in CIR.
///
/// There are three possible formats for this type:
///
///  - Identified and complete structs: unique name and a known body.
///  - Identified and incomplete structs: unique name and unkonwn body.
///  - Anonymous structs: no name and a known body.
///
/// Identified structs are uniqued by their name, and anonymous structs are
/// uniqued by their body. This means that two anonymous structs with the same
/// body will be the same type, and two identified structs with the same name
/// will be the same type. Attempting to build a struct with a existing name,
/// but a different body will result in an error.
///
/// A few examples:
///
/// ```mlir
///     !complete = !cir.struct<struct "complete" {!cir.int<u, 8>}>
///     !incomplete = !cir.struct<struct "incomplete" incomplete>
///     !anonymous = !cir.struct<struct {!cir.int<u, 8>}>
/// ```
///
/// Incomplete structs are mutable, meaning the can be later completed with a
/// body automatically updating in place every type in the code that uses the
/// incomplete struct. Mutability allows for recursive types to be represented,
/// meaning the struct can have members that refer to itself. This is useful for
/// representing recursive records and is implemented through a special syntax.
/// In the example below, the `Node` struct has a member that is a pointer to a
/// `Node` struct:
///
/// ```mlir
///     !struct = !cir.struct<struct "Node" {!cir.ptr<!cir.struct<struct
///     "Node">>}>
/// ```
class StructType
    : public Type::TypeBase<StructType, Type, detail::StructTypeStorage,
                            DataLayoutTypeInterface::Trait,
                            TypeTrait::IsMutable> {
  // FIXME(cir): migrate this type to Tablegen once mutable types are supported.
public:
  using Base::Base;
  using Base::getChecked;
  using Base::verify;

  static constexpr StringLiteral name = "cir.struct";

  enum RecordKind : uint32_t { Class, Union, Struct };

  /// Create a identified and complete struct type.
  static StructType get(MLIRContext *context, ArrayRef<Type> members,
                        StringAttr name, bool packed, RecordKind kind,
                        ASTRecordDeclInterface ast = {});
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, ArrayRef<Type> members,
                               StringAttr name, bool packed, RecordKind kind,
                               ASTRecordDeclInterface ast = {});

  /// Create a identified and incomplete struct type.
  static StructType get(MLIRContext *context, StringAttr name, RecordKind kind);
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, StringAttr name,
                               RecordKind kind);

  /// Create a anonymous struct type (always complete).
  static StructType get(MLIRContext *context, ArrayRef<Type> members,
                        bool packed, RecordKind kind,
                        ASTRecordDeclInterface ast = {});
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, ArrayRef<Type> members,
                               bool packed, RecordKind kind,
                               ASTRecordDeclInterface ast = {});

  /// Validate the struct about to be constructed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<Type> members, StringAttr name,
                              bool incomplete, bool packed,
                              StructType::RecordKind kind,
                              ASTRecordDeclInterface ast);

  // Parse/print methods.
  static constexpr StringLiteral getMnemonic() { return {"struct"}; }
  static Type parse(AsmParser &odsParser);
  void print(AsmPrinter &odsPrinter) const;

  // Accessors
  ASTRecordDeclInterface getAst() const;
  ArrayRef<Type> getMembers() const;
  StringAttr getName() const;
  StructType::RecordKind getKind() const;
  bool getIncomplete() const;
  bool getPacked() const;
  void dropAst();

  // Predicates
  bool isClass() const { return getKind() == RecordKind::Class; };
  bool isStruct() const { return getKind() == RecordKind::Struct; };
  bool isUnion() const { return getKind() == RecordKind::Union; };
  bool isComplete() const { return !isIncomplete(); };
  bool isIncomplete() const;

  // Utilities
  Type getLargestMember(const DataLayout &dataLayout) const;
  size_t getNumElements() const { return getMembers().size(); };
  std::string getKindAsStr() {
    switch (getKind()) {
    case RecordKind::Class:
      return "class";
    case RecordKind::Union:
      return "union";
    case RecordKind::Struct:
      return "struct";
    }
  }
  std::string getPrefixedName() {
    return getKindAsStr() + "." + getName().getValue().str();
  }

  /// Complete the struct type by mutating its members and attributes.
  void complete(ArrayRef<Type> members, bool packed,
                ASTRecordDeclInterface ast = {});

  /// DataLayoutTypeInterface methods.
  llvm::TypeSize getTypeSizeInBits(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const;
  uint64_t getABIAlignment(const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const;
  uint64_t getPreferredAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const;

  bool isLayoutIdentical(const StructType &other);

  // Utilities for lazily computing and cacheing data layout info.
private:
  // FIXME: currently opaque because there's a cycle if CIRTypes.types include
  // from CIRAttrs.h. The implementation operates in terms of StructLayoutAttr
  // instead.
  mutable mlir::Attribute layoutInfo;
  bool isPadded(const DataLayout &dataLayout) const;
  uint64_t getElementOffset(const DataLayout &dataLayout, unsigned idx) const;

  void computeSizeAndAlignment(const DataLayout &dataLayout) const;
};

} // namespace cir
} // namespace mlir

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

#endif // MLIR_DIALECT_CIR_IR_CIRTYPES_H_
