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
// TBAAScalarAttr
//===----------------------------------------------------------------------===//

def CIR_TBAAScalarAttr : CIR_Attr<"TBAAScalar", "tbaa_scalar", [], "TBAAAttr"> {
  let summary = "Describes a scalar type in TBAA with an identifier.";

  let parameters = (ins StringRefParameter<> : $id, CIR_AnyType : $type);

  let description = [{
    Define a TBAA scalar attribute.

    Example:
    ```mlir
    // CIR_TBAAScalarAttr
    #tbaa_scalar = #cir.tbaa_scalar<id = "int", type = !s32i>
    #tbaa_scalar1 = #cir.tbaa_scalar<id = "long long", type = !s64i>
    ```
    
    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_TBAATagAttr : CIR_Attr<"TBAATag", "tbaa_tag", [], "TBAAAttr"> {
  let parameters = (ins CIR_TBAAAttr
                    : $base, CIR_TBAAAttr
                    : $access, "int64_t"
                    : $offset);

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_TBAAStructAttr : CIR_Attr<"TBAAStruct", 
                                  "tbaa_struct", [], "TBAAAttr"> {
  let summary = "Describes a struct type in TBAA";

  let parameters = (ins CIR_StructType : $type);

  let description = [{
    Define a TBAA struct attribute.

    Example:
    ```mlir
    // CIR_TBAAStructAttr
    !ty_StructA = !cir.struct<struct "StructA" {!u16i, !u32i, !u16i, !u32i} #cir.record.decl.ast>
    !ty_StructS = !cir.struct<struct "StructS" {!u16i, !u32i} #cir.record.decl.ast>
    #tbaa_struct = #cir.tbaa_struct<type = !ty_StructA>
    #tbaa_struct1 = #cir.tbaa_struct<type = !ty_StructS>
    ```
    
    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def CIR_AnyTBAAAttr : AnyAttrOf<[
  CIR_TBAAAttr, 
  CIR_TBAAOmnipotentChar, 
  CIR_TBAAScalarAttr,
  CIR_TBAAStructAttr,
  CIR_TBAATagAttr
]>;
