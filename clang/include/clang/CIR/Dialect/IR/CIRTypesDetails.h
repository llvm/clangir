//===- CIRTypesDetails.h - Details of CIR dialect types ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementation details, such as storage structures, of
// CIR dialect types.
//
//===----------------------------------------------------------------------===//
#ifndef CIR_DIALECT_IR_CIRTYPESDETAILS_H
#define CIR_DIALECT_IR_CIRTYPESDETAILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/Hashing.h"

namespace cir {
namespace detail {

//===----------------------------------------------------------------------===//
// CIR RecordTypeStorage
//===----------------------------------------------------------------------===//

/// Type storage for CIR record types.
struct RecordTypeStorage : public mlir::TypeStorage {
  struct KeyTy {
    llvm::ArrayRef<mlir::Type> members;
    llvm::StringRef name;
    bool complete;
    bool packed;
    bool padded;
    RecordKind kind;
    ASTRecordDeclInterface ast;

    KeyTy(llvm::ArrayRef<mlir::Type> members, llvm::StringRef name,
          bool complete, bool packed, bool padded, RecordKind kind,
          ASTRecordDeclInterface ast)
        : members(members), name(name), complete(complete), packed(packed),
          padded(padded), kind(kind), ast(ast) {}
  };

  llvm::ArrayRef<mlir::Type> members;
  llvm::StringRef name;
  bool complete;
  bool packed;
  bool padded;
  RecordKind kind;
  ASTRecordDeclInterface ast;

  RecordTypeStorage(llvm::ArrayRef<mlir::Type> members, llvm::StringRef name,
                    bool complete, bool packed, bool padded, RecordKind kind,
                    ASTRecordDeclInterface ast)
      : members(members), name(name), complete(complete), packed(packed),
        padded(padded), kind(kind), ast(ast) {}

  KeyTy getAsKey() const {
    return KeyTy(members, name, complete, packed, padded, kind, ast);
  }

  bool operator==(const KeyTy &key) const {
    if (!name.empty())
      return (name == key.name) && (kind == key.kind);
    return (members == key.members) && (name == key.name) &&
           (complete == key.complete) && (packed == key.packed) &&
           (padded == key.padded) && (kind == key.kind) && (ast == key.ast);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (!key.name.empty())
      return llvm::hash_combine(key.name, key.kind);
    return llvm::hash_combine(key.members, key.complete, key.packed, key.padded,
                              key.kind, key.ast);
  }

  static RecordTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<RecordTypeStorage>()) RecordTypeStorage(
        allocator.copyInto(key.members), key.name, key.complete, key.packed,
        key.padded, key.kind, key.ast);
  }

  /// Mutates the members and attributes an identified record.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations. Anonymous records are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete record does not
  /// change the record.
  llvm::LogicalResult mutate(mlir::TypeStorageAllocator &allocator,
                             llvm::ArrayRef<mlir::Type> members, bool packed,
                             bool padded, ASTRecordDeclInterface ast) {
    // Anonymous records cannot mutate.
    if (name.empty())
      return llvm::failure();

    // Mutation of complete records are allowed if they change nothing.
    if (complete)
      return mlir::success((this->members == members) &&
                           (this->packed == packed) &&
                           (this->padded == padded) && (this->ast == ast));

    // Mutate incomplete record.
    this->members = allocator.copyInto(members);
    this->packed = packed;
    this->ast = ast;
    this->padded = padded;

    complete = true;
    return llvm::success();
  }
};

} // namespace detail
} // namespace cir

#endif // CIR_DIALECT_IR_CIRTYPESDETAILS_H
