//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NAMED_TUPLE_OPS
#define NAMED_TUPLE_OPS

include "mlir/Dialect/NamedTuple/IR/NamedTupleDialect.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class NamedTuple_Op<string mnemonic, list<Trait> traits = []>
    : Op<NamedTuple_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def NamedTuple_CastOp : NamedTuple_Op<"cast", [
      DeclareOpInterfaceMethods<CastOpInterface>,
      Pure,
    ]> {
  let summary = "WIP memref cast operation for NamedTuple hack";
  let description = [{
    The `named_tuple.cast` operation converts a memref of `named_tuple` to a 1D memref of `i8` of the same size to emulate later the struct element access.

    Example:

    ```mlir
    %0 = named_tuple.cast %alloca_0 : memref<!named_tuple.named_tuple<"s", [i32, i32]>> to memref<8xi8> loc(#loc6)
    ```
  }];

  let arguments = (ins AnyRankedOrUnrankedMemRef:$source);
  let results = (outs AnyRankedOrUnrankedMemRef:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

#endif // NAMED_TUPLE_OPS
