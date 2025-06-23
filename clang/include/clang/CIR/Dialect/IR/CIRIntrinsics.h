//===- CIRIntrinsics.h - CIR Intrinsic Function Handling ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a set of enums which allow processing of intrinsic
// functions. Values of these enum types are returned by
// Function::getIntrinsicID.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_INTRINSICS_H
#define LLVM_CLANG_CIR_DIALECT_INTRINSICS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/TypeSize.h"
#include <optional>
#include <string>
// #include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIRFPTypeInterface.h"

namespace mlir {
class Type;
class ModuleOp;
class MLIRContext;
} // namespace mlir

namespace llvm {
class StringRef;

namespace Intrinsic {
struct IITDescriptor; // No need to duplicate the definition here
} // namespace Intrinsic

} // namespace llvm
namespace cir {
class LLVMIntrinsicCallOp;
class FuncOp;
// FIXME: Unsure if we need a proper function type

namespace CIRIntrinsic {

// Abstraction for the arguments of the noalias intrinsics
static const int NoAliasScopeDeclScopeArg = 0;

// Intrinsic ID type. This is an opaque typedef to facilitate splitting up
// the enum into target-specific enums.
typedef unsigned ID;

enum IndependentIntrinsics : unsigned {
  not_intrinsic = 0, // Must be zero

// Get the intrinsic enums generated from Intrinsics.td
#define GET_INTRINSIC_ENUM_VALUES
#include "llvm/IR/IntrinsicEnums.inc"
#undef GET_INTRINSIC_ENUM_VALUES
};

// Simple descriptor struct that holds essential intrinsic information
// In order to build CIRIntrinsicCallOp
struct IntrinsicDescriptor {
  mlir::StringAttr name; // Mangled name attribute
  mlir::Type resultType; // Return type for the intrinsic
  ID id;                 // Original intrinsic ID (optional)

  // Basic constructor
  IntrinsicDescriptor(mlir::StringAttr name, mlir::Type resultType,
                      ID id = not_intrinsic)
      : name(name), resultType(resultType), id(id) {}

  // Default constructor for empty/invalid descriptors
  IntrinsicDescriptor()
      : name(nullptr), resultType(nullptr), id(not_intrinsic) {}

  // Check if descriptor is valid
  bool isValid() const { return name && resultType; }
};

/// Return the LLVM name for an intrinsic, such as "llvm.ppc.altivec.lvx".
/// Note, this version is for intrinsics with no overloads.  Use the other
/// version of getName if overloads are required.
llvm::StringRef getName(ID id);

/// Return the LLVM name for an intrinsic, without encoded types for
/// overloading, such as "llvm.ssa.copy".
llvm::StringRef getBaseName(ID id);

/// Return the LLVM name for an intrinsic, such as "llvm.ppc.altivec.lvx" or
/// "llvm.ssa.copy.p0s_s.1". Note, this version of getName supports overloads.
/// This is less efficient than the StringRef version of this function.  If no
/// overloads are required, it is safe to use this version, but better to use
/// the StringRef version. If one of the types is based on an unnamed type, a
/// function type will be computed. Providing FT will avoid this computation.
std::string getName(ID Id, llvm::ArrayRef<mlir::Type> Tys, mlir::ModuleOp M,
                    mlir::Type FT = nullptr);

/// Return the LLVM name for an intrinsic. This is a special version only to
/// be used by LLVMIntrinsicCopyOverloadedName. It only supports overloads
/// based on named types.
std::string getNameNoUnnamedTypes(ID Id, llvm::ArrayRef<mlir::Type> Tys);

/// Return the function type for an intrinsic.
// mlir::Type getType(mlir::MLIRContext &Context, ID id,
//                    llvm::ArrayRef<mlir::Type> Tys = {});

// Get both return type and parameter types in one call
mlir::Type getType(mlir::MLIRContext &Context, ID id,
                   llvm::ArrayRef<mlir::Type> Tys);

/// Returns true if the intrinsic can be overloaded.
bool isOverloaded(ID id); // NYI

ID lookupIntrinsicID(llvm::StringRef Name);

// FIXME: Uses table from LLVM, but we don't have it yet.
/// Return the attributes for an intrinsic.
//   AttributeList getAttributes(mlir::MLIRContext &C, ID id);
// this is also defined as:
// /// This defines the "Intrinsic::getAttributes(ID id)" method.
// #define GET_INTRINSIC_ATTRIBUTES
// #include "llvm/IR/IntrinsicImpl.inc"
// #undef GET_INTRINSIC_ATTRIBUTES

/// Look up the Function declaration of the intrinsic \p id in the Module
/// \p M. If it does not exist, add a declaration and return it. Otherwise,
/// return the existing declaration.
///
/// The \p Tys parameter is for intrinsics with overloaded types (e.g., those
/// using iAny, fAny, vAny, or pAny).  For a declaration of an overloaded
/// intrinsic, Tys must provide exactly one type for each overloaded type in
/// the intrinsic.
IntrinsicDescriptor getOrInsertDeclaration(mlir::ModuleOp M, ID id,
                                           llvm::ArrayRef<mlir::Type> Tys = {});

/// Look up the Function declaration of the intrinsic \p id in the Module
/// \p M and return it if it exists. Otherwise, return nullptr. This version
/// supports non-overloaded intrinsics.
IntrinsicDescriptor getDeclarationIfExists(const mlir::ModuleOp *M, ID id);

/// This version supports overloaded intrinsics.
IntrinsicDescriptor getDeclarationIfExists(mlir::ModuleOp M, ID id,
                                           llvm::ArrayRef<mlir::Type> Tys,
                                           mlir::Type FT = nullptr);

/// Map a Clang builtin name to an intrinsic ID.
ID getIntrinsicForClangBuiltin(llvm::StringRef TargetPrefix,
                               llvm::StringRef BuiltinName);

/// Map a MS builtin name to an intrinsic ID.
ID getIntrinsicForMSBuiltin(llvm::StringRef TargetPrefix,
                            llvm::StringRef BuiltinName);

// FIXME: Uses table from LLVM, but we don't have it yet.
//  /// Returns true if the intrinsic ID is for one of the "Constrained
//  /// Floating-Point Intrinsics".
//  bool isConstrainedFPIntrinsic(ID QID);

// /// Returns true if the intrinsic ID is for one of the "Constrained
// /// Floating-Point Intrinsics" that take rounding mode metadata.
// bool hasConstrainedFPRoundingModeOperand(ID QID);

/// Return the IIT table descriptor for the specified intrinsic into an array
/// of IITDescriptors.
void getIntrinsicInfoTableEntries(
    ID id, llvm::SmallVectorImpl<llvm::Intrinsic::IITDescriptor> &T);

enum MatchIntrinsicTypesResult {
  MatchIntrinsicTypes_Match = 0,
  MatchIntrinsicTypes_NoMatchRet = 1,
  MatchIntrinsicTypes_NoMatchArg = 2,
};

/// Match the specified function type with the type constraints specified by
/// the .td file. If the given type is an overloaded type it is pushed to the
/// ArgTys vector.
///
/// Returns false if the given type matches with the constraints, true
/// otherwise.
MatchIntrinsicTypesResult
matchIntrinsicSignature(FuncOp FTy,
                        llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> &Infos,
                        llvm::SmallVectorImpl<mlir::Type> &ArgTys);

/// Verify if the intrinsic has variable arguments. This method is intended to
/// be called after all the fixed arguments have been matched first.
///
/// This method returns true on error.
bool matchIntrinsicVarArg(
    bool isVarArg, llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> &Infos);

/// Gets the type arguments of an intrinsic call by matching type contraints
/// specified by the .td file. The overloaded types are pushed into the
/// AgTys vector.
///
/// Returns false if the given ID and function type combination is not a
/// valid intrinsic call.
bool getIntrinsicSignature(ID, mlir::Type FT,
                           llvm::SmallVectorImpl<mlir::Type> &ArgTys);

/// Same as previous, but accepts a Function instead of ID and FunctionType.
bool getIntrinsicSignature(FuncOp F, llvm::SmallVectorImpl<mlir::Type> &ArgTys);

// Checks if the intrinsic name matches with its signature and if not
// returns the declaration with the same signature and remangled name.
// An existing GlobalValue with the wanted name but with a wrong prototype
// or of the wrong kind will be renamed by adding ".renamed" to the name.
std::optional<LLVMIntrinsicCallOp> remangleIntrinsicFunction(FuncOp F);

} // namespace CIRIntrinsic
} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_INTRINSICS_H