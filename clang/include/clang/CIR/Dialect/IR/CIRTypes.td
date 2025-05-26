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
include "clang/CIR/Dialect/IR/CIRTypeConstraints.td"
include "clang/CIR/Interfaces/ASTAttrInterfaces.td"
include "clang/CIR/Interfaces/CIRTypeInterfaces.td"
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
    fundamental integer types. Said types are: `signed char`, `short`, `int`,
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
    /// Return true if this is a fundamental integer type (i.e. signed or
    /// unsigned integer types whose bit width is 8, 16, 32, or 64).
    bool isFundamental() const {
      return isFundamentalIntType(*this);
    }
    bool isSignedFundamental() const {
      return isFundamentalSIntType(*this);
    }
    bool isUnsignedFundamental() const {
      return isFundamentalUIntType(*this);
    }

    /// Returns a minimum bitwidth of cir::IntType
    static unsigned minBitwidth() { return 1; }
    /// Returns a maximum bitwidth of cir::IntType
    static unsigned maxBitwidth() { return 128; }
  }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// FloatType
//===----------------------------------------------------------------------===//

class CIR_FloatType<string name, string mnemonic> : CIR_Type<name, mnemonic, [
  DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
  DeclareTypeInterfaceMethods<CIR_FPTypeInterface>
]>;

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
    underlying floating-point format is the IEEE-754 binary64 format.
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
    type that corresponds to this format.
  }];

  let parameters = (ins AnyTypeOf<[CIR_Double, CIR_FP80, CIR_FP128],
    "expects !cir.double, !cir.fp80 or !cir.fp128">:$underlying);

  let assemblyFormat = [{
    `<` $underlying `>`
  }];
}

//===----------------------------------------------------------------------===//
// ComplexType
//===----------------------------------------------------------------------===//

def CIR_ComplexType : CIR_Type<"Complex", "complex",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR complex type";
  let description = [{
    CIR type that represents a C complex number. `cir.complex` models the C type
    `T _Complex`.

    The type models complex values, per C99 6.2.5p11. It supports the C99
    complex float types as well as the GCC integer complex extensions.

    The parameter `elementType` gives the type of the real and imaginary part of
    the complex number. `elementType` must be either a CIR integer type or a CIR
    floating-point type.
  }];

  let parameters = (ins CIR_AnyIntOrFloatType:$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $elementType `>`
  }];

  let extraClassDeclaration = [{
    bool isFloatingPointComplex() const {
      return isAnyFloatingPointType(getElementType());
    }

    bool isIntegerComplex() const {
      return mlir::isa<cir::IntType>(getElementType());
    }
  }];
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
    template <typename ...Types>
    bool isPtrTo() const {
      return mlir::isa< Types... >(getPointee());
    }

    bool isVoidPtr() const {
      return isPtrTo<cir::VoidType>();
    }

    template <typename ...Types>
    bool isPtrToPtrTo() const {
      if (auto ptrType = mlir::dyn_cast<cir::PointerType>(getPointee()))
        return ptrType.isPtrTo<Types...>();
      return false;
    }

    bool isPtrTo(mlir::Type type) const {
      return getPointee() == type;
    }

    bool isPtrToPtrTo(mlir::Type type) const {
      if (auto ptrType = mlir::dyn_cast<cir::PointerType>(getPointee()))
        return ptrType.isPtrTo(type);
      return false;
    }
  }];
}

//===----------------------------------------------------------------------===//
// CIR_DataMemberType
//===----------------------------------------------------------------------===//

def CIR_DataMemberType : CIR_Type<"DataMember", "data_member",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR type that represents pointer-to-data-member type in C++";
  let description = [{
    `cir.data_member` models the pointer-to-data-member type in C++. Values of
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
//===----------------------------------------------------------------------===//

def CIR_BoolType : CIR_Type<"Bool", "bool",
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

  let parameters = (ins "mlir::Type":$elementType, "uint64_t":$size);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$elementType, "uint64_t":$size
    ), [{
        return $_get(elementType.getContext(), elementType, size);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $elementType `x` $size `>`
  }];
}

//===----------------------------------------------------------------------===//
// VectorType (fixed size)
//===----------------------------------------------------------------------===//

def CIR_VectorType : CIR_Type<"Vector", "vector",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "CIR vector type";
  let description = [{
    The `!cir.vector` type represents a fixed-size, one-dimensional vector.
    It takes two parameters: the element type and the number of elements.

    Syntax:

    ```mlir
    vector-type ::= !cir.vector<element-type x size>
    element-type ::= float-type | integer-type | pointer-type
    ```

    The `element-type` must be a scalar CIR type. Zero-sized vectors are not
    allowed. The `size` must be a positive integer.

    Examples:

    ```mlir
    !cir.vector<!cir.int<u, 8> x 4>
    !cir.vector<!cir.float x 2>
    ```
  }];

  let parameters = (ins
    CIR_VectorElementType:$elementType,
    "uint64_t":$size
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "mlir::Type":$elementType, "uint64_t":$size
    ), [{
        return $_get(elementType.getContext(), elementType, size);
    }]>,
  ];

  let assemblyFormat = [{
    `<` $elementType `x` $size `>`
  }];

  let genVerifyDecl = 1;
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

//===----------------------------------------------------------------------===//
// RecordType
//
// The base type for all RecordDecls.
//===----------------------------------------------------------------------===//

def CIR_RecordKind : I32EnumAttr<"RecordKind", "Record kind", [
  I32EnumAttrCase<"Class", 0, "Class">,
  I32EnumAttrCase<"Union", 1, "Union">,
  I32EnumAttrCase<"Struct", 2, "Struct">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::cir";
}

def CIR_RecordType : CIR_Type<"Record", "record",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, MutableType]> {
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

    Examples:

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
    StringRefParameter<"mlir::StringRef">:$name,
    "bool":$complete,
    "bool":$packed,
    "bool":$padded,
    CIR_RecordKind:$kind,
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

    bool isClass() const { return getKind() == RecordKind::Class; };
    bool isStruct() const { return getKind() == RecordKind::Struct; };
    bool isUnion() const { return getKind() == RecordKind::Union; };
    bool isComplete() const;
    bool isIncomplete() const { return !isComplete(); };

    bool isAnonymous() const {
      return getName().empty();
    }

    void dropAst();

    mlir::Type getLargestMember(const mlir::DataLayout &dataLayout) const;
    size_t getNumElements() const { return getMembers().size(); };

    std::string getPrefixedName() const {
      return (stringifyRecordKind(getKind()) + "." + getName()).str();
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
  CIR_RecordType, CIR_ExceptionType, CIR_AnyFloatType, CIR_ComplexType
]>;

#endif // MLIR_CIR_DIALECT_CIR_TYPES
