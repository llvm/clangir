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
include "clang/CIR/Interfaces/ASTAttrInterfaces.td"
include "clang/CIR/Interfaces/CIRFPTypeInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// CIR Types
//===----------------------------------------------------------------------===//

class CIR_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<CIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

def CIR_IntType : CIR_Type<"Int", "int",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    CIR type that represents integer types with arbitrary precision.

    Those integer types that are directly available in C/C++ standard are called
    primitive integer types. Said types are: `signed char`, `short`, `int`,
    `long`, `long long`, and their unsigned variations.
  }];
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Return true if this is a signed integer type.
    bool isSigned() const { return getIsSigned(); }
    /// Return true if this is an unsigned integer type.
    bool isUnsigned() const { return !getIsSigned(); }
    /// Return type alias.
    std::string getAlias() const {
      return (isSigned() ? 's' : 'u') + std::to_string(getWidth()) + 'i';
    };
    /// Return true if this is a primitive integer type (i.e. signed or unsigned
    /// integer types whose bit width is 8, 16, 32, or 64).
    bool isPrimitive() const {
      return isValidPrimitiveIntBitwidth(getWidth());
    }
    bool isSignedPrimitive() const {
      return isPrimitive() && isSigned();
    }

    /// Returns a minimum bitwidth of cir::IntType
    static unsigned minBitwidth() { return 1; }
    /// Returns a maximum bitwidth of cir::IntType
    static unsigned maxBitwidth() { return 128; }

    /// Returns true if cir::IntType that represents a primitive integer type
    /// can be constructed from the provided bitwidth.
    static bool isValidPrimitiveIntBitwidth(unsigned width) {
      return width == 8 || width == 16 || width == 32 || width == 64;
    }
  }];
  let genVerifyDecl = 1;
}

// Constraints

// Unsigned integer type of a specific width.
class UInt<int width>
  : Type<And<[
        CPred<"::mlir::isa<::cir::IntType>($_self)">,
        CPred<"::mlir::cast<::cir::IntType>($_self).isUnsigned()">,
        CPred<"::mlir::cast<::cir::IntType>($_self).getWidth() == " # width>
        ]>, width # "-bit unsigned integer", "::cir::IntType">,
    BuildableType<
      "cir::IntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

def UInt1  : UInt<1>;
def UInt8  : UInt<8>;
def UInt16 : UInt<16>;
def UInt32 : UInt<32>;
def UInt64 : UInt<64>;

// Signed integer type of a specific width.
class SInt<int width>
  : Type<And<[
        CPred<"::mlir::isa<::cir::IntType>($_self)">,
        CPred<"::mlir::cast<::cir::IntType>($_self).isSigned()">,
        CPred<"::mlir::cast<::cir::IntType>($_self).getWidth() == " # width>
        ]>, width # "-bit signed integer", "::cir::IntType">,
    BuildableType<
      "cir::IntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}

def SInt1  : SInt<1>;
def SInt8  : SInt<8>;
def SInt16 : SInt<16>;
def SInt32 : SInt<32>;
def SInt64 : SInt<64>;

def PrimitiveUInt
    : AnyTypeOf<[UInt8, UInt16, UInt32, UInt64], "primitive unsigned int",
                "::cir::IntType">;
def PrimitiveSInt
    : AnyTypeOf<[SInt8, SInt16, SInt32, SInt64], "primitive signed int",
                "::cir::IntType">;
def PrimitiveInt
    : AnyTypeOf<[UInt8, UInt16, UInt32, UInt64, SInt8, SInt16, SInt32, SInt64],
                "primitive int", "::cir::IntType">;

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

class CIR_FloatType<string name, string mnemonic>
    : CIR_Type<name, mnemonic,
          [
            DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
            DeclareTypeInterfaceMethods<CIRFPTypeInterface>,
          ]> {}

def CIR_Single : CIR_FloatType<"Single", "float"> {
  let summary = "CIR single-precision float type";
  let description = [{
    Floating-point type that represents the `float` type in C/C++. Its
    underlying floating-point format is the IEEE-754 binary32 format.
  }];
}

def CIR_Double : CIR_FloatType<"Double", "double"> {
  let summary = "CIR double-precision float type";
  let description = [{
    Floating-point type that represents the `double` type in C/C++. Its
    underlying floating-point format is the IEEE-754 binar64 format.
  }];
}

def CIR_FP16 : CIR_FloatType<"FP16", "f16"> {
  let summary = "CIR type that represents IEEE-754 binary16 format";
  let description = [{
    Floating-point type that represents the IEEE-754 binary16 format.
  }];
}

def CIR_BFloat16 : CIR_FloatType<"BF16", "bf16"> {
  let summary = "CIR type that represents";
  let description = [{
    Floating-point type that represents the bfloat16 format.
  }];
}

def CIR_FP80 : CIR_FloatType<"FP80", "f80"> {
  let summary = "CIR type that represents x87 80-bit floating-point format";
  let description = [{
    Floating-point type that represents the x87 80-bit floating-point format.
  }];
}

def CIR_FP128 : CIR_FloatType<"FP128", "f128"> {
  let summary = "CIR type that represents IEEEquad 128-bit floating-point format";
  let description = [{
    Floating-point type that represents the IEEEquad 128-bit floating-point format.
  }];
}

def CIR_LongDouble : CIR_FloatType<"LongDouble", "long_double"> {
  let summary = "CIR extended-precision float type";
  let description = [{
    Floating-point type that represents the `long double` type in C/C++.

    The underlying floating-point format of a long double value depends on the
    implementation. The `underlying` parameter specifies the CIR floating-point
    type that corresponds to this format. For now, it can only be either
    `!cir.double` or `!cir.fp80`.
  }];

  let parameters = (ins "mlir::Type":$underlying);

  let assemblyFormat = [{
    `<` $underlying `>`
  }];

  let genVerifyDecl = 1;
}

// Constraints

def CIR_AnyFloat: AnyTypeOf<[CIR_Single, CIR_Double, CIR_FP80, CIR_FP128, CIR_LongDouble,
    CIR_FP16, CIR_BFloat16]>;
def CIR_AnyIntOrFloat: AnyTypeOf<[CIR_AnyFloat, CIR_IntType]>;

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

def CIR_ComplexType : CIR_Type<"Complex", "complex",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR complex type";
  let description = [{
    CIR type that represents a C complex number. `cir.complex` models the C type
    `T _Complex`.

    The parameter `elementTy` gives the type of the real and imaginary part of
    the complex number. `elementTy` must be either a CIR integer type or a CIR
    floating-point type.
  }];

  let parameters = (ins "mlir::Type":$elementTy);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementTy), [{
      return $_get(elementTy.getContext(), elementTy);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $elementTy `>`
  }];

  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

def CIR_PointerType : CIR_Type<"Pointer", "ptr",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR pointer type";
  let description = [{
    `CIR.ptr` is a type returned by any op generating a pointer in C++.
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    // FIXME(cir): Currently unable to directly use AddressSpaceAttr because of
    // cyclic dep. Workaround with the top type and verifier.
    OptionalParameter<"mlir::Attribute">:$addrSpace
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$pointee,
      CArg<"mlir::Attribute", "{}">:$addrSpace), [{
      return $_get(pointee.getContext(), pointee, addrSpace);
    }]>,
    TypeBuilder<(ins
      "mlir::Type":$pointee,
      CArg<"mlir::Attribute", "{}">:$addrSpace), [{
      return $_get($_ctxt, pointee, addrSpace);
    }]>
  ];

  let assemblyFormat = [{
    `<` $pointee ( `,` `addrspace` `(`
      custom<PointerAddrSpace>($addrSpace)^
    `)` )? `>`
  }];

  let genVerifyDecl = 1;

  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    bool isVoidPtr() const {
      return mlir::isa<cir::VoidType>(getPointee());
    }
  }];
}

//===----------------------------------------------------------------------===//
// DataMemberType
//===----------------------------------------------------------------------===//

def CIR_DataMemberType : CIR_Type<"DataMember", "data_member",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR type that represents pointer-to-data-member type in C++";
  let description = [{
    `cir.member_ptr` models the pointer-to-data-member type in C++. Values of
    this type are essentially offsets of the pointed-to member within one of
    its containing record.
  }];

  let parameters = (ins "mlir::Type":$memberTy,
                        "cir::RecordType":$clsTy);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$memberTy, "cir::RecordType":$clsTy
    ), [{
      return $_get(memberTy.getContext(), memberTy, clsTy);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $memberTy `in` $clsTy `>`
  }];
}

//===----------------------------------------------------------------------===//
// BoolType
//
// An alternative here is to represent bool as mlir::i1, but let's be more
// generic.
//
//===----------------------------------------------------------------------===//

def CIR_BoolType :
    CIR_Type<"Bool", "bool",
             [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR bool type";
  let description = [{
    `cir.bool` represent's C++ bool type.
  }];
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def CIR_ArrayType : CIR_Type<"Array", "array",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR array type";
  let description = [{
    `CIR.array` represents C/C++ constant arrays.
  }];

  let parameters = (ins "mlir::Type":$eltType, "uint64_t":$size);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$eltType, "uint64_t":$size
    ), [{
        return $_get(eltType.getContext(), eltType, size);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $eltType `x` $size `>`
  }];
}

//===----------------------------------------------------------------------===//
// VectorType (fixed size)
//===----------------------------------------------------------------------===//

def CIR_VectorType : CIR_Type<"Vector", "vector",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR vector type";
  let description = [{
    `cir.vector' represents fixed-size vector types.  The parameters are the
    element type and the number of elements.
  }];

  let parameters = (ins "mlir::Type":$eltType, "uint64_t":$size);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$eltType, "uint64_t":$size
    ), [{
        return $_get(eltType.getContext(), eltType, size);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $eltType `x` $size `>`
  }];
}

//===----------------------------------------------------------------------===//
// FuncType
//===----------------------------------------------------------------------===//

def CIR_FuncType : CIR_Type<"Func", "func"> {
  let summary = "CIR function type";
  let description = [{
    The `!cir.func` is a function type. It consists of an optional return type,
    a list of parameter types and can optionally be variadic.

    Example:

    ```mlir
    !cir.func<()>
    !cir.func<() -> !bool>
    !cir.func<(!s8i, !s8i)>
    !cir.func<(!s8i, !s8i) -> !s32i>
    !cir.func<(!s32i, ...) -> !s32i>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"mlir::Type">:$inputs,
                        OptionalParameter<"mlir::Type">:$optionalReturnType,
                        "bool":$varArg);
  // Use a custom parser to handle argument types with variadic elipsis.
  let assemblyFormat = [{
    `<` custom<FuncTypeParams>($inputs, $varArg)  (`->` $optionalReturnType^)? `>`
  }];

  let builders = [
    // Create a FuncType, converting the return type from C-style to
    // MLIR-style.  If the given return type is `cir::VoidType`, ignore it
    // and create the FuncType with no return type, which is how MLIR
    // represents function types.
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<mlir::Type>":$inputs, "mlir::Type":$returnType,
      CArg<"bool", "false">:$isVarArg), [{
        return $_get(returnType.getContext(), inputs,
                     mlir::isa<cir::VoidType>(returnType) ? nullptr
                                                          : returnType,
                     isVarArg);
    }]>
  ];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns whether the function is variadic.
    bool isVarArg() const { return getVarArg(); }

    /// Returns the `i`th input operand type. Asserts if out of bounds.
    mlir::Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Returns the number of arguments to the function.
    unsigned getNumInputs() const { return getInputs().size(); }

    /// Get the C-style return type of the function, which is !cir.void if the
    /// function returns nothing and the actual return type otherwise.
    mlir::Type getReturnType() const;

    /// Get the MLIR-style return type of the function, which is an empty
    /// ArrayRef if the function returns nothing and a single-element ArrayRef
    /// with the actual return type otherwise.
    llvm::ArrayRef<mlir::Type> getReturnTypes() const;

    /// Does the fuction type return nothing?
    bool hasVoidReturn() const;

    /// Returns a clone of this function type with the given argument
    /// and result types.
    FuncType clone(mlir::TypeRange inputs, mlir::TypeRange results) const;
  }];
}

//===----------------------------------------------------------------------===//
// MethodType
//===----------------------------------------------------------------------===//

def CIR_MethodType : CIR_Type<"Method", "method",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "CIR type that represents C++ pointer-to-member-function type";
  let description = [{
    `cir.method` models the pointer-to-member-function type in C++. The layout
    of this type is ABI-dependent.
  }];

  let parameters = (ins "cir::FuncType":$memberFuncTy,
                        "cir::RecordType":$clsTy);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "cir::FuncType":$memberFuncTy, "cir::RecordType":$clsTy
    ), [{
      return $_get(memberFuncTy.getContext(), memberFuncTy, clsTy);
    }]>,
  ];

  let assemblyFormat = [{
    `<` qualified($memberFuncTy) `in` $clsTy `>`
  }];
}

//===----------------------------------------------------------------------===//
// Exception info type
//
// By introducing an exception info type, exception related operations can be
// more descriptive.
//
// This basically wraps a uint8_t* and a uint32_t
//
//===----------------------------------------------------------------------===//

def CIR_ExceptionType : CIR_Type<"ExceptionInfo", "exception"> {
  let summary = "CIR exception info";
  let description = [{
    In presence of an inflight exception, this type holds all specific
    information for an exception: the associated type id, and the exception
    object pointer. These are materialzed from this type through other
    specific operations.
  }];
}

//===----------------------------------------------------------------------===//
// Void type
//===----------------------------------------------------------------------===//

def CIR_VoidType : CIR_Type<"Void", "void"> {
  let summary = "CIR void type";
  let description = [{
    The `!cir.void` type represents the C/C++ `void` type.
  }];
  let extraClassDeclaration = [{
    /// Returns a clone of this type with the given context.
    std::string getAlias() const { return "void"; };
  }];
}

// Constraints

// Pointer to void
def VoidPtr : Type<
    And<[
      CPred<"::mlir::isa<::cir::PointerType>($_self)">,
      CPred<"::mlir::isa<::cir::VoidType>("
            "::mlir::cast<::cir::PointerType>($_self).getPointee())">,
    ]>, "void*">,
    BuildableType<
      "cir::PointerType::get($_builder.getContext(),"
      "cir::VoidType::get($_builder.getContext()))"> {
}

// Pointer to a primitive int, float or double
def PrimitiveIntOrFPPtr : Type<
    And<[
      CPred<"::mlir::isa<::cir::PointerType>($_self)">,
      CPred<"::mlir::isa<::cir::IntType, ::cir::SingleType,"
            "::cir::DoubleType>("
            "::mlir::cast<::cir::PointerType>($_self).getPointee())">,
    ]>, "{int,void}*"> {
}

def ComplexPtr : Type<
    And<[
      CPred<"::mlir::isa<::cir::PointerType>($_self)">,
      CPred<"::mlir::isa<::cir::ComplexType>("
        "::mlir::cast<::cir::PointerType>($_self).getPointee())">,
    ]>, "!cir.complex*"> {
}

// Pointer to record
def RecordPtr : Type<
    And<[
      CPred<"::mlir::isa<::cir::PointerType>($_self)">,
      CPred<"::mlir::isa<::cir::RecordType>("
            "::mlir::cast<::cir::PointerType>($_self).getPointee())">
    ]>, "!cir.record*"> {
}

// Pointer to exception info
def ExceptionPtr : Type<
    And<[
      CPred<"::mlir::isa<::cir::PointerType>($_self)">,
      CPred<"::mlir::isa<::cir::ExceptionInfoType>("
            "::mlir::cast<::cir::PointerType>($_self).getPointee())">
    ]>, "!cir.eh_info*">,
    BuildableType<
      "cir::PointerType::get($_builder.getContext(),"
      "cir::ExceptionInfoType::get($_builder.getContext()))"> {
}

// Vector of integral type
def IntegerVector : Type<
    And<[
      CPred<"::mlir::isa<::cir::VectorType>($_self)">,
      CPred<"::mlir::isa<::cir::IntType>("
            "::mlir::cast<::cir::VectorType>($_self).getEltType())">,
      CPred<"::mlir::cast<::cir::IntType>("
            "::mlir::cast<::cir::VectorType>($_self).getEltType())"
            ".isPrimitive()">
    ]>, "!cir.vector of !cir.int"> {
}

// Vector of signed integral type
def SignedIntegerVector : Type<
    And<[
      CPred<"::mlir::isa<::cir::VectorType>($_self)">,
      CPred<"::mlir::isa<::cir::IntType>("
            "::mlir::cast<::cir::VectorType>($_self).getEltType())">,
      CPred<"::mlir::cast<::cir::IntType>("
            "::mlir::cast<::cir::VectorType>($_self).getEltType())"
            ".isSignedPrimitive()">
    ]>, "!cir.vector of !cir.int"> {
}

// Vector of Float type
def FPVector : Type<
    And<[
      CPred<"::mlir::isa<::cir::VectorType>($_self)">,
      CPred<"::mlir::isa<::cir::SingleType, ::cir::DoubleType>("
            "::mlir::cast<::cir::VectorType>($_self).getEltType())">,
    ]>, "!cir.vector of !cir.fp"> {
}

// Constraints
def CIR_AnyIntOrVecOfInt: AnyTypeOf<[CIR_IntType, IntegerVector]>;
def CIR_AnySignedIntOrVecOfSignedInt: AnyTypeOf<
                                        [PrimitiveSInt, SignedIntegerVector]>;
def CIR_AnyFloatOrVecOfFloat: AnyTypeOf<[CIR_AnyFloat, FPVector]>;

// Pointer to Arrays
def ArrayPtr : Type<
    And<[
      CPred<"::mlir::isa<::cir::PointerType>($_self)">,
      CPred<"::mlir::isa<::cir::ArrayType>("
            "::mlir::cast<::cir::PointerType>($_self).getPointee())">,
    ]>, "!cir.ptr<!cir.array>"> {
}

// Pointer to functions
def FuncPtr : Type<
    And<[
      CPred<"::mlir::isa<::cir::PointerType>($_self)">,
      CPred<"::mlir::isa<::cir::FuncType>("
            "::mlir::cast<::cir::PointerType>($_self).getPointee())">,
    ]>, "!cir.ptr<!cir.func>"> {
}

//===----------------------------------------------------------------------===//
// RecordType
//
// The base type for all RecordDecls.
//===----------------------------------------------------------------------===//

def CIR_RecordType : CIR_Type<"Record", "record",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
      MutableType,
    ]> {
  let summary = "CIR record type";
  let description = [{
    Each unique clang::RecordDecl is mapped to a `cir.record` and any object in
    C/C++ that has a struct or class type will have a `cir.record` in CIR.

    There are three possible formats for this type:

     - Identified and complete records: unique name and a known body.
     - Identified and incomplete records: unique name and unknown body.
     - Anonymous records: no name and a known body.

    Identified records are uniqued by their name, and anonymous records are
    uniqued by their body. This means that two anonymous records with the same
    body will be the same type, and two identified records with the same name
    will be the same type. Attempting to build a record with an existing name,
    but a different body will result in an error.

    A few examples:

    ```mlir
        !complete = !cir.record<struct {!cir.int<u, 8>}>
        !incomplete = !cir.record<struct "incomplete" incomplete>
        !anonymous = !cir.record<struct {!cir.int<u, 8>}>
    ```

    Incomplete records are mutable, meaning they can be later completed with a
    body automatically updating in place every type in the code that uses the
    incomplete record. Mutability allows for recursive types to be represented,
    meaning the record can have members that refer to itself. This is useful for
    representing recursive records and is implemented through a special syntax.
    In the example below, the `Node` record has a member that is a pointer to a
    `Node` record:

    ```mlir
        !s = !cir.record<struct "Node" {!cir.ptr<!cir.record<struct "Node">>}>
    ```
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"mlir::Type">:$members,
    OptionalParameter<"mlir::StringAttr">:$name,
    "bool":$complete,
    "bool":$packed,
    "bool":$padded,
    "RecordType::RecordKind":$kind,
    OptionalParameter<"ASTRecordDeclInterface">:$ast
  );

  // StorageClass is defined in C++ for mutability.
  let storageClass = "RecordTypeStorage";
  let genStorageClass = 0;

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    // Create an identified and complete record type.
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "mlir::StringAttr":$name,
      "bool":$packed,
      "bool":$padded,
      "RecordKind":$kind,
      CArg<"ASTRecordDeclInterface", "{}">:$ast
    ), [{
      return $_get($_ctxt, members, name, /*complete=*/true, packed, padded,
                       kind, ast);    
    }]>,

    // Create an identified and incomplete record type.
    TypeBuilder<(ins
      "mlir::StringAttr":$name,
      "RecordKind":$kind
    ), [{
      return $_get($_ctxt, /*members=*/llvm::ArrayRef<Type>{}, name,
                         /*complete=*/false, /*packed=*/false,
                         /*padded=*/false, kind,
                         /*ast=*/ASTRecordDeclInterface{});    
    }]>,

    // Create an anonymous record type (always complete).
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members,
      "bool":$packed,
      "bool":$padded,
      "RecordKind":$kind,
      CArg<"ASTRecordDeclInterface", "{}">:$ast
    ), [{
      return $_get($_ctxt, members, mlir::StringAttr{}, /*complete=*/true,
                      packed, padded, kind, ast);
    }]>];

  let extraClassDeclaration = [{
    using Base::verifyInvariants;

    enum RecordKind : uint32_t { Class, Union, Struct };

    bool isClass() const { return getKind() == RecordKind::Class; };
    bool isStruct() const { return getKind() == RecordKind::Struct; };
    bool isUnion() const { return getKind() == RecordKind::Union; };
    bool isComplete() const;
    bool isIncomplete() const { return !isComplete(); };

    void dropAst();

    mlir::Type getLargestMember(const mlir::DataLayout &dataLayout) const;
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
      llvm_unreachable("Invalid value for RecordType::getKind()");
    }
    std::string getPrefixedName() {
      return getKindAsStr() + "." + getName().getValue().str();
    }

    void complete(llvm::ArrayRef<mlir::Type> members, bool packed,
              bool isPadded, ASTRecordDeclInterface ast = {});

    uint64_t getElementOffset(const mlir::DataLayout &dataLayout,
              unsigned idx) const;

    bool isLayoutIdentical(const RecordType &other);

  // Utilities for computing data layout info
  private:
    unsigned computeStructSize(const mlir::DataLayout &dataLayout) const;
    unsigned computeUnionSize(const mlir::DataLayout &dataLayout) const;
    uint64_t computeStructAlignment(const mlir::DataLayout &dataLayout) const;
    uint64_t computeUnionAlignment(const mlir::DataLayout &dataLayout) const;
  public:
  }];

  let hasCustomAssemblyFormat = 1;
}

// Note CIRRecordType is used instead of CIR_RecordType
// because of tablegen conflicts.
def CIRRecordType : Type<
  CPred<"::mlir::isa<::cir::RecordType>($_self)">, "CIR record type">;

//===----------------------------------------------------------------------===//
// Global type constraints
//===----------------------------------------------------------------------===//

def CIR_AnyType : AnyTypeOf<[
  CIR_IntType, CIR_PointerType, CIR_DataMemberType, CIR_MethodType,
  CIR_BoolType, CIR_ArrayType, CIR_VectorType, CIR_FuncType, CIR_VoidType,
  CIR_RecordType, CIR_ExceptionType, CIR_AnyFloat, CIR_FP16, CIR_BFloat16,
  CIR_ComplexType
]>;

#endif // MLIR_CIR_DIALECT_CIR_TYPES
