//===- CIRAttrs.td - CIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_ATTRS
#define MLIR_CIR_DIALECT_CIR_ATTRS

include "mlir/Dialect/CIR/IR/CIRDialect.td"
include "mlir/Dialect/CIR/IR/CIRTypes.td"

//===----------------------------------------------------------------------===//
// CIR Attrs
//===----------------------------------------------------------------------===//

class CIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def CstArrayAttr : CIR_Attr<"CstArray", "cst_array"> {
  let summary = "A constant array from ArrayAttr or StringRefAttr";
  let description = [{
    An CIR array attribute is an array of literals of the specified attr types.
  }];

  // `$type` is the `self` type of the attribute (i.e. the type of the
  // Attribute itself).
  //
  // `arrayAttr` is the actual attribute array with elements for this constant
  // array, there's yet no need to own these elements.
  //
  // TODO: create a trait for ArrayAttrOrStringAttr value instead of relying
  // on verifier.
  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        "Attribute":$value);

  // Define a custom builder for the type; that removes the need to pass
  // in an MLIRContext instance, as it can be infered from the `type`.
  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::cir::ArrayType":$type,
                                        "Attribute":$value), [{
      return $_get(type.getContext(), type, value);
    }]>
  ];

  // Printing and parsing available in CIRDialect.cpp

  // Enable verifier.
  let genVerifyDecl = 1;
}

#endif // MLIR_CIR_DIALECT_CIR_ATTRS

