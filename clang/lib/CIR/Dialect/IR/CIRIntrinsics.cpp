//===-- CIRIntrinsics.cpp - Intrinsic Function Handling ------------*- C++
//-*-===//
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

using namespace cir;

/// Table of string intrinsic names indexed by enum value.
#define GET_INTRINSIC_NAME_TABLE
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_NAME_TABLE

llvm::StringRef CIRIntrinsic::getBaseName(ID id) {
  assert(id < num_intrinsics && "Invalid intrinsic ID!");

  llvm::StringRef fullName = IntrinsicNameTable[IntrinsicNameOffsetTable[id]];
  // The format returned is llvm.<name>, so we need to skip the "llvm." to
  // represent
  //  our intrinsic call ops.
  return fullName.substr(5);
}

llvm::StringRef CIRIntrinsic::getName(ID id) {
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  assert(!CIRIntrinsic::isOverloaded(id) &&
         "This version of getName does not support overloading");
  return getBaseName(id);
}

std::string CIRIntrinsic::getName(ID Id, llvm::ArrayRef<mlir::Type> Tys,
                                  mlir::ModuleOp M, mlir::Type FT) {
  llvm_unreachable("GetIntrinsic Overloading NYI");
  // assert(M && "We need to have a Module");
  // return getIntrinsicNameImpl(Id, Tys, M, FT, true);
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
  llvm_unreachable("Intrinsic Mangling NYI");
  return {};
}

static std::string getIntrinsicNameImpl(cir::CIRIntrinsic::ID Id,
                                        llvm::ArrayRef<mlir::Type> Tys,
                                        mlir::ModuleOp M, mlir::Type FT,
                                        bool EarlyModuleCheck) {

  llvm_unreachable("NYI: getIntrinsicNameImpl for CIR");
  return {};
}

CIRIntrinsic::IntrinsicDescriptor
CIRIntrinsic::getOrInsertDeclaration(mlir::ModuleOp M, ID id,
                                     llvm::ArrayRef<mlir::Type> Tys) {
  LLVMIntrinsicCallOp res;

  // There can never be multiple globals with the same name of different types,
  // because intrinsics must be a specific type.
  mlir::Type ResultTy = CIRIntrinsic::getType(*M->getContext(), id, Tys);

  // 2. Generate the mangled name for the intrinsic
  llvm::StringRef Name =
      Tys.empty()
          ? getName(id)
          : getName(id, Tys, M); // FIXME: Need to strip the llvm. prefix

  mlir::StringAttr nameAttr = mlir::StringAttr::get(M->getContext(), Name);

  return CIRIntrinsic::IntrinsicDescriptor(nameAttr, ResultTy, id);
}

//(cir)TODO: Most IIT Descriptor helpers should be common to Codegen to avoid
// duplication.
/// IIT_Info - These are enumerators that describe the entries returned by the
/// getIntrinsicInfoTableEntries function.
///
/// Defined in Intrinsics.td.
enum IIT_Info {
#define GET_INTRINSIC_IITINFO
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_IITINFO
};

static void DecodeIITType(
    unsigned &NextElt, llvm::ArrayRef<unsigned char> Infos, IIT_Info LastInfo,
    llvm::SmallVectorImpl<llvm::Intrinsic::IITDescriptor> &OutputTable) {
  using namespace llvm::Intrinsic;

  bool IsScalableVector = (LastInfo == IIT_SCALABLE_VEC);

  IIT_Info Info = IIT_Info(Infos[NextElt++]);
  unsigned StructElts = 2;

  switch (Info) {
  case IIT_Done:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Void, 0));
    return;
  case IIT_VARARG:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::VarArg, 0));
    return;
  case IIT_MMX:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::MMX, 0));
    return;
  case IIT_AMX:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::AMX, 0));
    return;
  case IIT_TOKEN:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Token, 0));
    return;
  case IIT_METADATA:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Metadata, 0));
    return;
  case IIT_F16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Half, 0));
    return;
  case IIT_BF16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::BFloat, 0));
    return;
  case IIT_F32:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Float, 0));
    return;
  case IIT_F64:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Double, 0));
    return;
  case IIT_F128:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Quad, 0));
    return;
  case IIT_PPCF128:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::PPCQuad, 0));
    return;
  case IIT_I1:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 1));
    return;
  case IIT_I2:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 2));
    return;
  case IIT_I4:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 4));
    return;
  case IIT_AARCH64_SVCOUNT:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::AArch64Svcount, 0));
    return;
  case IIT_I8:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 8));
    return;
  case IIT_I16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 16));
    return;
  case IIT_I32:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 32));
    return;
  case IIT_I64:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 64));
    return;
  case IIT_I128:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 128));
    return;
  case IIT_V1:
    OutputTable.push_back(IITDescriptor::getVector(1, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V2:
    OutputTable.push_back(IITDescriptor::getVector(2, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V3:
    OutputTable.push_back(IITDescriptor::getVector(3, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V4:
    OutputTable.push_back(IITDescriptor::getVector(4, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V6:
    OutputTable.push_back(IITDescriptor::getVector(6, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V8:
    OutputTable.push_back(IITDescriptor::getVector(8, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V10:
    OutputTable.push_back(IITDescriptor::getVector(10, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V16:
    OutputTable.push_back(IITDescriptor::getVector(16, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V32:
    OutputTable.push_back(IITDescriptor::getVector(32, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V64:
    OutputTable.push_back(IITDescriptor::getVector(64, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V128:
    OutputTable.push_back(IITDescriptor::getVector(128, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V256:
    OutputTable.push_back(IITDescriptor::getVector(256, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V512:
    OutputTable.push_back(IITDescriptor::getVector(512, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_V1024:
    OutputTable.push_back(IITDescriptor::getVector(1024, IsScalableVector));
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  case IIT_EXTERNREF:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer, 10));
    return;
  case IIT_FUNCREF:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer, 20));
    return;
  case IIT_PTR:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer, 0));
    return;
  case IIT_ANYPTR: // [ANYPTR addrspace]
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Pointer, Infos[NextElt++]));
    return;
  case IIT_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Argument, ArgInfo));
    return;
  }
  case IIT_EXTEND_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::ExtendArgument, ArgInfo));
    return;
  }
  case IIT_TRUNC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::TruncArgument, ArgInfo));
    return;
  }
  case IIT_HALF_VEC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::HalfVecArgument, ArgInfo));
    return;
  }
  case IIT_ONE_THIRD_VEC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::OneThirdVecArgument, ArgInfo));
    return;
  }
  case IIT_ONE_FIFTH_VEC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::OneFifthVecArgument, ArgInfo));
    return;
  }
  case IIT_ONE_SEVENTH_VEC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::OneSeventhVecArgument, ArgInfo));
    return;
  }
  case IIT_SAME_VEC_WIDTH_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::SameVecWidthArgument, ArgInfo));
    return;
  }
  case IIT_VEC_OF_ANYPTRS_TO_ELT: {
    unsigned short ArgNo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    unsigned short RefNo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::VecOfAnyPtrsToElt, ArgNo, RefNo));
    return;
  }
  case IIT_EMPTYSTRUCT:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Struct, 0));
    return;
  case IIT_STRUCT9:
    ++StructElts;
    [[fallthrough]];
  case IIT_STRUCT8:
    ++StructElts;
    [[fallthrough]];
  case IIT_STRUCT7:
    ++StructElts;
    [[fallthrough]];
  case IIT_STRUCT6:
    ++StructElts;
    [[fallthrough]];
  case IIT_STRUCT5:
    ++StructElts;
    [[fallthrough]];
  case IIT_STRUCT4:
    ++StructElts;
    [[fallthrough]];
  case IIT_STRUCT3:
    ++StructElts;
    [[fallthrough]];
  case IIT_STRUCT2: {
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Struct, StructElts));

    for (unsigned i = 0; i != StructElts; ++i)
      DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  }
  case IIT_SUBDIVIDE2_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Subdivide2Argument, ArgInfo));
    return;
  }
  case IIT_SUBDIVIDE4_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::Subdivide4Argument, ArgInfo));
    return;
  }
  case IIT_VEC_ELEMENT: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::VecElementArgument, ArgInfo));
    return;
  }
  case IIT_SCALABLE_VEC: {
    DecodeIITType(NextElt, Infos, Info, OutputTable);
    return;
  }
  case IIT_VEC_OF_BITCASTS_TO_INT: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(
        IITDescriptor::get(IITDescriptor::VecOfBitcastsToInt, ArgInfo));
    return;
  }
  }
  llvm_unreachable("unhandled");
}

#define GET_INTRINSIC_GENERATOR_GLOBAL
#include "llvm/IR/IntrinsicImpl.inc"
#undef GET_INTRINSIC_GENERATOR_GLOBAL

void CIRIntrinsic::getIntrinsicInfoTableEntries(
    ID id, llvm::SmallVectorImpl<llvm::Intrinsic::IITDescriptor> &T) {
  static_assert(sizeof(IIT_Table[0]) == 2,
                "Expect 16-bit entries in IIT_Table");
  // Check to see if the intrinsic's type was expressible by the table.
  uint16_t TableVal = IIT_Table[id - 1];

  // Decode the TableVal into an array of IITValues.
  llvm::SmallVector<unsigned char> IITValues;
  llvm::ArrayRef<unsigned char> IITEntries;
  unsigned NextElt = 0;
  if (TableVal >> 15) {
    // This is an offset into the IIT_LongEncodingTable.
    IITEntries = IIT_LongEncodingTable;

    // Strip sentinel bit.
    NextElt = TableVal & 0x7fff;
  } else {
    // If the entry was encoded into a single word in the table itself, decode
    // it from an array of nibbles to an array of bytes.
    do {
      IITValues.push_back(TableVal & 0xF);
      TableVal >>= 4;
    } while (TableVal);

    IITEntries = IITValues;
    NextElt = 0;
  }

  // Okay, decode the table into the output vector of IITDescriptors.
  DecodeIITType(NextElt, IITEntries, IIT_Done, T);
  while (NextElt != IITEntries.size() && IITEntries[NextElt] != 0)
    DecodeIITType(NextElt, IITEntries, IIT_Done, T);
}

// Cir requires us to specify signedness for integer types. This function
// determines if the intrinsic is signed or unsigned based on the intrinsic ID.
// This is needed as the IITDescriptor
//  does not contain metadata about signedness.
static bool isSigned(unsigned IID) {
  using namespace llvm::Intrinsic;

  switch (IID) {
  default:
    return true;
  case llvm::Intrinsic::x86_rdtsc:
  case llvm::Intrinsic::x86_rdtscp:
  case llvm::Intrinsic::ctlz:
    // Add more cases as needed
    return false;
  }
}

static mlir::Type
DecodeFixedType(llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> &Infos,
                llvm::ArrayRef<mlir::Type> Tys, mlir::MLIRContext &Context,
                unsigned IID) {
  using namespace llvm::Intrinsic;

  IITDescriptor D = Infos.front();
  Infos = Infos.slice(1);
  bool IsSignedInt = isSigned(IID);

  switch (D.Kind) {
  case IITDescriptor::Void:
    return cir::VoidType::get(&Context);
  case IITDescriptor::VarArg:
    return cir::VoidType::get(&Context);
  case IITDescriptor::MMX:
    return cir::VectorType::get(&Context, cir::IntType::get(&Context, 64, true),
                                1);
  case IITDescriptor::AMX:
    llvm_unreachable("AMX intrinsic mapping type NYI");
  case IITDescriptor::Token:
    llvm_unreachable("Token intrinsic mapping type NYI");
  case IITDescriptor::Metadata:
    llvm_unreachable("Metadata intrinsic mapping type NYI");
  case IITDescriptor::Half:
    return cir::FP16Type::get(&Context);
  case IITDescriptor::BFloat:
    return cir::FP128Type::get(&Context);
  case IITDescriptor::Float:
    return cir::SingleType::get(&Context);
  case IITDescriptor::Double:
    return cir::DoubleType::get(&Context);
  case IITDescriptor::Quad:
    return cir::FP128Type::get(&Context);
  case IITDescriptor::PPCQuad:
    llvm_unreachable("PPC Quad type NYI");
  case IITDescriptor::AArch64Svcount:
    llvm_unreachable(
        "AArch64 SVCount type NYI - this is a target extension type");
  case IITDescriptor::Integer:
    // FIXME:  Not sure if we should default to signed or unsigned here as
    // IITDescriptor does not contain relevant metadata.
    return cir::IntType::get(&Context, D.Integer_Width, IsSignedInt);

  case IITDescriptor::Vector:
    return cir::VectorType::get(&Context,
                                DecodeFixedType(Infos, Tys, Context, IID),
                                D.Vector_Width.getKnownMinValue());
  case IITDescriptor::Pointer: {
    mlir::Type i8Type =
        cir::IntType::get(&Context, 8, true); // or whatever represents i8
    mlir::IntegerAttr addrSpaceAttr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(&Context, 32), D.Pointer_AddressSpace);
    return cir::PointerType::get(i8Type, addrSpaceAttr);
  }
  case IITDescriptor::Struct: {
    llvm::SmallVector<mlir::Type, 8> memberTypes;

    // Collect all the member types from the descriptor
    for (unsigned i = 0, e = D.Struct_NumElements; i != e; ++i)
      memberTypes.push_back(DecodeFixedType(Infos, Tys, Context, IID));

    // Create and return the RecordType
    return cir::RecordType::get(
        &Context, memberTypes,
        mlir::StringAttr(), // Name of the record, empty for anonymous
        false,
        true,                   // Most structs are padded for alignment
        cir::RecordType::Struct // Record kind (struct/class/union)
    );
  }
    //   case IITDescriptor::Argument:
    //     return Tys[D.getArgumentNumber()];
    //   case IITDescriptor::ExtendArgument: {
    //     Type *Ty = Tys[D.getArgumentNumber()];
    //     if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    //       return VectorType::getExtendedElementVectorType(VTy);

    //     return IntegerType::get(Context, 2 *
    //     cast<IntegerType>(Ty)->getBitWidth());
    //   }
    //   case IITDescriptor::TruncArgument: {
    //     Type *Ty = Tys[D.getArgumentNumber()];
    //     if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    //       return VectorType::getTruncatedElementVectorType(VTy);

    //     IntegerType *ITy = cast<IntegerType>(Ty);
    //     assert(ITy->getBitWidth() % 2 == 0);
    //     return IntegerType::get(Context, ITy->getBitWidth() / 2);
    //   }
    //   case IITDescriptor::Subdivide2Argument:
    //   case IITDescriptor::Subdivide4Argument: {
    //     Type *Ty = Tys[D.getArgumentNumber()];
    //     VectorType *VTy = dyn_cast<VectorType>(Ty);
    //     assert(VTy && "Expected an argument of Vector Type");
    //     int SubDivs = D.Kind == IITDescriptor::Subdivide2Argument ? 1 : 2;
    //     return VectorType::getSubdividedVectorType(VTy, SubDivs);
    //   }
    //   case IITDescriptor::HalfVecArgument:
    //     return VectorType::getHalfElementsVectorType(
    //         cast<VectorType>(Tys[D.getArgumentNumber()]));
    //   case IITDescriptor::OneThirdVecArgument:
    //   case IITDescriptor::OneFifthVecArgument:
    //   case IITDescriptor::OneSeventhVecArgument:
    //     return VectorType::getOneNthElementsVectorType(
    //         cast<VectorType>(Tys[D.getArgumentNumber()]),
    //         3 + (D.Kind - IITDescriptor::OneThirdVecArgument) * 2);
    //   case IITDescriptor::SameVecWidthArgument: {
    //     Type *EltTy = DecodeFixedType(Infos, Tys, Context);
    //     Type *Ty = Tys[D.getArgumentNumber()];
    //     if (auto *VTy = dyn_cast<VectorType>(Ty))
    //       return VectorType::get(EltTy, VTy->getElementCount());
    //     return EltTy;
    //   }
    //   case IITDescriptor::VecElementArgument: {
    //     Type *Ty = Tys[D.getArgumentNumber()];
    //     if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    //       return VTy->getElementType();
    //     llvm_unreachable("Expected an argument of Vector Type");
    //   }
    //   case IITDescriptor::VecOfBitcastsToInt: {
    //     Type *Ty = Tys[D.getArgumentNumber()];
    //     VectorType *VTy = dyn_cast<VectorType>(Ty);
    //     assert(VTy && "Expected an argument of Vector Type");
    //     return VectorType::getInteger(VTy);
    //   }
    //   case IITDescriptor::VecOfAnyPtrsToElt:
    //     // Return the overloaded type (which determines the pointers address
    //     space) return Tys[D.getOverloadArgNumber()];
    //   }
    llvm_unreachable("unhandled");
  }
  return {};
}

bool CIRIntrinsic::isOverloaded(ID id) {
  return false; // NYI
}

// TODO: Lil placeholder for now, we will need to re-implement this later
mlir::Type CIRIntrinsic::getType(mlir::MLIRContext &Context, ID id,
                                 llvm::ArrayRef<mlir::Type> Tys) {
  llvm::SmallVector<llvm::Intrinsic::IITDescriptor, 8> Table;
  CIRIntrinsic::getIntrinsicInfoTableEntries(id, Table);

  // Process the table once to get both return and parameter types
  llvm::ArrayRef<llvm::Intrinsic::IITDescriptor> TableRef = Table;

  // First descriptor is the return type
  mlir::Type ResultTy = DecodeFixedType(TableRef, Tys, Context, id);

  // NOTE: In theory we could deal with the arg types here,
  // But I don't see the point given that we don't care about strict typing for
  // Intrinsic Ops...

  //  if (!ArgTys.empty() && ArgTys.back()->isVoidTy()) {
  //   ArgTys.pop_back();
  //   return FunctionType::get(ResultTy, ArgTys, true);
  return ResultTy;
}
