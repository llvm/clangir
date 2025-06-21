//===-- CIRIntrinsics.cpp - Intrinsic Function Handling ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions required for supporting intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRIntrinsics.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace cir;

/// Table of string intrinsic names indexed by enum value.
#define GET_INTRINSIC_NAME_TABLE
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_NAME_TABLE

llvm::StringRef Intrinsic::getBaseName(ID id) {
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  return IntrinsicNameTable[IntrinsicNameOffsetTable[id]];
}

llvm::StringRef Intrinsic::getName(ID id) {
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  assert(!Intrinsic::isOverloaded(id) &&
         "This version of getName does not support overloading");
  return getBaseName(id);
}

/// Returns a stable mangling for the type specified for use in the name
/// mangling scheme used by 'any' types in intrinsic signatures.  The mangling
/// of named types is simply their name.  Manglings for unnamed types consist
/// of a prefix ('p' for pointers, 'a' for arrays, 'f_' for functions)
/// combined with the mangling of their component types.  A vararg function
/// type will have a suffix of 'vararg'.  Since function types can contain
/// other function types, we close a function type mangling with suffix 'f'
/// which can't be confused with it's prefix.  This ensures we don't have
/// collisions between two unrelated function types. Otherwise, you might
/// parse ffXX as f(fXX) or f(fX)X.  (X is a placeholder for any other type.)
/// The HasUnnamedType boolean is set if an unnamed type was encountered,
/// indicating that extra care must be taken to ensure a unique name.
static std::string getMangledTypeStr(mlir::Type Ty, bool &HasUnnamedType) {
  llvm_unreachable("NYI");
  std::string Result;
//   if (PointerType *PTyp = dyn_cast<PointerType>(Ty)) {
//     Result += "p" + utostr(PTyp->getAddressSpace());
//   } else if (ArrayType *ATyp = dyn_cast<ArrayType>(Ty)) {
//     Result += "a" + utostr(ATyp->getNumElements()) +
//               getMangledTypeStr(ATyp->getElementType(), HasUnnamedType);
//   } else if (StructType *STyp = dyn_cast<StructType>(Ty)) {
//     if (!STyp->isLiteral()) {
//       Result += "s_";
//       if (STyp->hasName())
//         Result += STyp->getName();
//       else
//         HasUnnamedType = true;
//     } else {
//       Result += "sl_";
//       for (auto *Elem : STyp->elements())
//         Result += getMangledTypeStr(Elem, HasUnnamedType);
//     }
//     // Ensure nested structs are distinguishable.
//     Result += "s";
//   } else if (FunctionType *FT = dyn_cast<FunctionType>(Ty)) {
//     Result += "f_" + getMangledTypeStr(FT->getReturnType(), HasUnnamedType);
//     for (size_t i = 0; i < FT->getNumParams(); i++)
//       Result += getMangledTypeStr(FT->getParamType(i), HasUnnamedType);
//     if (FT->isVarArg())
//       Result += "vararg";
//     // Ensure nested function types are distinguishable.
//     Result += "f";
//   } else if (VectorType *VTy = dyn_cast<VectorType>(Ty)) {
//     ElementCount EC = VTy->getElementCount();
//     if (EC.isScalable())
//       Result += "nx";
//     Result += "v" + utostr(EC.getKnownMinValue()) +
//               getMangledTypeStr(VTy->getElementType(), HasUnnamedType);
//   } else if (TargetExtType *TETy = dyn_cast<TargetExtType>(Ty)) {
//     Result += "t";
//     Result += TETy->getName();
//     for (Type *ParamTy : TETy->type_params())
//       Result += "_" + getMangledTypeStr(ParamTy, HasUnnamedType);
//     for (unsigned IntParam : TETy->int_params())
//       Result += "_" + utostr(IntParam);
//     // Ensure nested target extension types are distinguishable.
//     Result += "t";
//   } else if (Ty) {
//     switch (Ty->getTypeID()) {
//     default:
//       llvm_unreachable("Unhandled type");
//     case Type::VoidTyID:
//       Result += "isVoid";
//       break;
//     case Type::MetadataTyID:
//       Result += "Metadata";
//       break;
//     case Type::HalfTyID:
//       Result += "f16";
//       break;
//     case Type::BFloatTyID:
//       Result += "bf16";
//       break;
//     case Type::FloatTyID:
//       Result += "f32";
//       break;
//     case Type::DoubleTyID:
//       Result += "f64";
//       break;
//     case Type::X86_FP80TyID:
//       Result += "f80";
//       break;
//     case Type::FP128TyID:
//       Result += "f128";
//       break;
//     case Type::PPC_FP128TyID:
//       Result += "ppcf128";
//       break;
//     case Type::X86_AMXTyID:
//       Result += "x86amx";
//       break;
//     case Type::IntegerTyID:
//       Result += "i" + utostr(cast<IntegerType>(Ty)->getBitWidth());
//       break;
//     }
//   }
  return Result;
}

//TODO: This takes care of overloading, let's focus on the non-overloaded
static std::string getIntrinsicNameImpl(Intrinsic::ID Id, llvm::ArrayRef<mlir::Type> Tys,
                                        mlir::ModuleOp M, mlir::Type FT,
                                        bool EarlyModuleCheck) {

  llvm_unreachable("NYI: getIntrinsicNameImpl for CIR");
  std::string Result;

//   assert(Id < Intrinsic::num_intrinsics && "Invalid intrinsic ID!");
//   assert((Tys.empty() || Intrinsic::isOverloaded(Id)) &&
//          "This version of getName is for overloaded intrinsics only");
//   (void)EarlyModuleCheck;
//   assert((!EarlyModuleCheck || M ||
//           !any_of(Tys, [](Type *T) { return isa<PointerType>(T); })) &&
//          "Intrinsic overloading on pointer types need to provide a Module");
//   bool HasUnnamedType = false;
//   std::string Result(Intrinsic::getBaseName(Id));
//   for (Type *Ty : Tys)
//     Result += "." + getMangledTypeStr(Ty, HasUnnamedType);
//   if (HasUnnamedType) {
//     assert(M && "unnamed types need a module");
//     if (!FT)
//       FT = Intrinsic::getType(M->getContext(), Id, Tys);
//     else
//       assert((FT == Intrinsic::getType(M->getContext(), Id, Tys)) &&
//              "Provided FunctionType must match arguments");
//     return M->getUniqueIntrinsicName(Result, Id, FT);
//   }
  return Result;
}

LLVMIntrinsicCallOp Intrinsic::getOrInsertDeclaration(mlir::ModuleOp M, ID id,
                                            llvm::ArrayRef<mlir::Type> Tys) {
  // There can never be multiple globals with the same name of different types,
  // because intrinsics must be a specific type.
    auto *FT = Intrinsic::getType(*M.getContext(), id, Tys);

    //From our module we should get a global function with the name of the intrinsic. (if not overloaded)
    // return cast<Function>(
    //   M->getOrInsertFunction(
    //        Tys.empty() ? getName(id) : getName(id, Tys, M, FT), FT)
    //       .getCallee());
  return {};
}
