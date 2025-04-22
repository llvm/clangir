//===----------------------------------------------------------------------===//
// TBAAAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAAttr : CIR_Attr<"TBAA", "tbaa", []> {
  let summary = "CIR dialect TBAA base attribute";
}

//===----------------------------------------------------------------------===//
// TBAAOmnipotentCharAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAOmnipotentChar
    : CIR_Attr<"TBAAOmnipotentChar", "tbaa_omnipotent_char", [], "TBAAAttr"> {
  let summary = "Describes a special scalar type, the omnipotent char type.";
}

//===----------------------------------------------------------------------===//
// TBAAVTablePointerAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAVTablePointerAttr
    : CIR_Attr<"TBAAVTablePointer", "tbaa_vptr", [], "TBAAAttr"> {
  let summary = "Describes a special scalar type, the vtable pointer type.";
  let parameters = (ins CIR_AnyType:$type);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// TBAAScalarAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAScalarAttr : CIR_Attr<"TBAAScalar", "tbaa_scalar", [], "TBAAAttr"> {
  let summary = "Describes a scalar type in TBAA with an identifier.";

  let parameters = (ins StringRefParameter<>:$id, 
                        CIR_AnyType:$type, 
                        OptionalParameter<"cir::TBAAScalarAttr">:$parent);

  let description = [{
    Define a TBAA scalar attribute.
    The optional `parent` attribute is used to describe the parent type of the 
    scalar type. If the `parent` is null or omitted, the parent type is the 
    `omnipotent char` type.

    Example:
    ```mlir
    // CIR_TBAAScalarAttr
    #tbaa_scalar = #cir.tbaa_scalar<id = "int", type = !s32i>
    #tbaa_scalar1 = #cir.tbaa_scalar<id = "long long", type = !s64i>

    #tbaa_scalar2 = #cir.tbaa_scalar<id = "any pointer", type = !cir.ptr<!s32i>>
    #tbaa_scalar3 = #cir.tbaa_scalar<id = "p1 int", type = !cir.ptr<!s32i>, 
                                     parent = #tbaa_scalar2>
    ```
    
    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let builders = [
    AttrBuilder<(ins "llvm::StringRef":$id, 
                     "mlir::Type":$type), [{
      return $_get($_ctxt, id, type, /*parent =*/ nullptr);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_TBAATagAttr : CIR_Attr<"TBAATag", "tbaa_tag", [], "TBAAAttr"> {
  let parameters = (ins CIR_TBAAAttr
                    : $base, CIR_TBAAAttr
                    : $access, "int64_t"
                    : $offset);

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_TBAAMemberAttr : CIR_Attr<"TBAAMember", "tbaa_member", []> {
  let summary = "Attribute representing a member of a TBAA structured type.";
  let parameters = (ins "TBAAAttr":$type_desc,
                        "int64_t":$offset);
  let description = [{
    Define a TBAA struct attribute.

    Example:
    ```mlir
    !rec_StructS = !cir.record<struct "StructS" {!u16i, !u32i} #cir.record.decl.ast>
    #tbaa_scalar = #cir.tbaa_scalar<id = "int", type = !s32i>
    #tbaa_scalar1 = #cir.tbaa_scalar<id = "short", type = !s16i>
    #tbaa_struct = #cir.tbaa_struct<id = "_ZTS7StructS", members = {<#tbaa_scalar1, 0>, <#tbaa_scalar, 4>}>
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` params `>`";
}

def CIR_TBAAMemberAttrArray : ArrayRefParameter<"TBAAMemberAttr"> {
  let summary = "Array of TBAAMemberAttr attributes.";
  let printer = [{
    $_printer << '{';
    llvm::interleaveComma($_self, $_printer, [&](TBAAMemberAttr attr) {
        $_printer.printStrippedAttrOrType(attr);
    });
    $_printer << '}';
  }];
  let parser = [{
    [&]() -> llvm::FailureOr<llvm::SmallVector<TBAAMemberAttr>> {
        using Result = llvm::SmallVector<TBAAMemberAttr>;
        if ($_parser.parseLBrace())
            return mlir::failure();
        llvm::FailureOr<Result> result = mlir::FieldParser<Result>::parse($_parser);
        if (failed(result))
            return mlir::failure();
        if ($_parser.parseRBrace())
            return mlir::failure();
        return result;
    }()
  }];
}

def CIR_TBAAStructAttr : CIR_Attr<"TBAAStruct",
                                  "tbaa_struct", [], "TBAAAttr"> {
  let summary = "Describes a struct type in TBAA";

  let parameters = (ins StringRefParameter<> : $id,
                        CIR_TBAAMemberAttrArray:$members);

  let description = [{
    Define a TBAA struct attribute.

    Example:
    ```mlir
    !rec_StructS = !cir.record<struct "StructS" {!u16i, !u32i} #cir.record.decl.ast>
    #tbaa_scalar = #cir.tbaa_scalar<id = "int", type = !s32i>
    #tbaa_scalar1 = #cir.tbaa_scalar<id = "short", type = !s16i>
    // CIR_TBAAStructAttr
    #tbaa_struct = #cir.tbaa_struct<id = "_ZTS7StructS", members = {<#tbaa_scalar1, 0>, <#tbaa_scalar, 4>}>
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_AnyTBAAAttr : AnyAttrOf<[
  CIR_TBAAAttr, 
  CIR_TBAAOmnipotentChar,
  CIR_TBAAVTablePointerAttr,
  CIR_TBAAScalarAttr,
  CIR_TBAAStructAttr,
  CIR_TBAATagAttr
]>;
