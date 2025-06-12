//===-- CIRStdOps.td - CIR standard library ops ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Defines ops representing standard library calls
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRSTDOPS
#define LLVM_CLANG_CIR_DIALECT_IR_CIRSTDOPS

class CIRStdOp<string functionName, dag args, dag res, list<Trait> traits = []>:
    CIR_Op<"std." # functionName, traits> {
  string funcName = functionName;

  let arguments = !con((ins FlatSymbolRefAttr:$original_fn), args);

  let summary = "std::" # functionName # "()";
  let results = res;

  let extraClassDeclaration = [{
    static constexpr unsigned getNumArgs() {
      return }] # !size(args) # [{;
    }
    static llvm::StringRef getFunctionName() {
      return "}] # functionName # [{";
    }
  }];

  string argsAssemblyFormat = !interleave(
    !foreach(
      name,
      !foreach(i, !range(!size(args)), !getdagname(args, i)),
      !strconcat("$", name, " `:` type($", name, ")")
    ), " `,` "
  );

  string resultAssemblyFormat = !if(
    !empty(res),
    "",
    " `->` type($" # !getdagname(res, 0) # ")"
  );

  let assemblyFormat = !strconcat("`(` ", argsAssemblyFormat,
                                  " `,` $original_fn `)`", resultAssemblyFormat,
                                  " attr-dict");
}

def StdFindOp : CIRStdOp<"find",
  (ins CIR_AnyType:$first, CIR_AnyType:$last, CIR_AnyType:$pattern),
  (outs CIR_AnyType:$result),
  [AllTypesMatch<["first", "last", "result"]>]>;
def IterBeginOp: CIRStdOp<"begin",
  (ins CIR_AnyType:$container),
  (outs CIR_AnyType:$result)>;
def IterEndOp: CIRStdOp<"end",
  (ins CIR_AnyType:$container),
  (outs CIR_AnyType:$result)>;

#endif
