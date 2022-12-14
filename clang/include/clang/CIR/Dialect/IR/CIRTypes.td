//===- CIRTypes.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_TYPES
#define MLIR_CIR_DIALECT_CIR_TYPES

include "clang/CIR/Dialect/IR/CIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// CIR Types
//===----------------------------------------------------------------------===//

class CIR_Type<string name, string typeMnemonic> : TypeDef<CIR_Dialect,
                                                           name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def CIR_PointerType :
    CIR_Type<"Pointer", "ptr"> {

  let summary = "CIR pointer type";
  let description = [{
    `CIR.ptr` is a type returned by any op generating a pointer in C++.
  }];

  let parameters = (ins "mlir::Type":$pointee);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// BoolType
//
// An alternative here is to represent bool as mlir::i1, but let's be more
// generic.
//
//===----------------------------------------------------------------------===//
def CIR_BoolType :
    CIR_Type<"Bool", "bool"> {

  let summary = "CIR bool type";
  let description = [{
    `cir.bool` represent's C++ bool type.
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// StructType
//
// The base type for all RecordDecls.
//
//===----------------------------------------------------------------------===//

def CIR_StructType : CIR_Type<"Struct", "struct"> {

  let summary = "CIR struct type";
  let description = [{
    Each unique clang::RecordDecl is mapped to a `cir.struct` and any object in
    C/C++ that has a struct type will have a `cir.struct` in CIR.
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::Type", "members">:$members,
    "mlir::StringAttr":$typeName,
    "bool":$body,
    "Optional<::mlir::cir::ASTRecordDeclAttr>":$ast
  );

  let builders = [
    TypeBuilder<(ins
      "ArrayRef<mlir::Type>":$members, "StringRef":$typeName,
      "bool":$body
    ), [{
      auto id = mlir::StringAttr::get(context, typeName);
      auto sTy = StructType::get(context, members, id, body, std::nullopt);
      return sTy;
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
  public:
    void dropAst();
  }];

  let extraClassDefinition = [{
    void $cppClass::dropAst() {
      getImpl()->ast = std::nullopt;
    }
  }];
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def CIR_ArrayType :
    CIR_Type<"Array", "array"> {

  let summary = "CIR array type";
  let description = [{
    `CIR.array` represents C/C++ constant arrays.
  }];

  let parameters = (ins "mlir::Type":$eltType, "uint64_t":$size);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// One type to bind them all
//===----------------------------------------------------------------------===//

def CIR_AnyCIRType : AnyTypeOf<[CIR_PointerType, CIR_BoolType, CIR_StructType,
                                CIR_ArrayType]>;

#endif // MLIR_CIR_DIALECT_CIR_TYPES
