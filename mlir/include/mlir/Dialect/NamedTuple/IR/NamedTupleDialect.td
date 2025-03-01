//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NAMED_TUPLE_DIALECT
#define NAMED_TUPLE_DIALECT

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// NamedTuple dialect definition.
//===----------------------------------------------------------------------===//

def NamedTuple_Dialect : Dialect {
  let name = "named_tuple";
  let summary = "NamedTuple dialect providing a tuple type with member names";
  let cppNamespace = "::mlir::named_tuple";
  let useDefaultTypePrinterParser = 1;
}

#endif // NAMED_TUPLE_DIALECT
