#ifndef MLIR_CIR_TRANSFORMS_STD_FUNCS
#define MLIR_CIR_TRANSFORMS_STD_FUNCS

include "mlir/IR/EnumAttr.td"

class CIRStdOp<string functionName, list<Trait> traits = []>:
    CIR_Op<"std." # functionName, traits> {
  string funcName = functionName;
  
  let arguments = (ins FlatSymbolRefAttr:$original_fn,
                       Variadic<CIR_AnyType>:$operands);

  let summary = "std::" # functionName # "()";
  let results = (outs CIR_AnyType:$result);

  let assemblyFormat = [{
    `(`
      $original_fn `,`
      $operands `:` type($operands)
    `)` `->` type($result) attr-dict
  }];
  let hasVerifier = 0;
}

multiclass CIRStdOpMulticlass<string functionName, int funcId,
                              list<Trait> traits = []> {
  def "" : CIRStdOp<functionName, traits>;
  
  def EnumEntry : I32EnumAttrCase<
    // Capitalize function name as enum entries.
    !toupper(!substr(functionName, 0, 1)) # !substr(functionName, 1),
    funcId,
    functionName
  >;
}

defm StdFindOp : CIRStdOpMulticlass<"find", 1, [SameFirstSecondOperandAndResultType]>;
defm IterBeginOp: CIRStdOpMulticlass<"begin", 2>;
defm IterEndOp: CIRStdOpMulticlass<"end", 3>;

def StdFuncID : I32EnumAttr<"StdFuncID", "maps standard function names to IDs", [
  StdFindOpEnumEntry,
  IterBeginOpEnumEntry,
  IterEndOpEnumEntry,
]> {
  let cppNamespace = "::cir";
}

#endif
