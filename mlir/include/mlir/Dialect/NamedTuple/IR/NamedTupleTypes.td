//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef NAMED_TUPLE_TYPES
#define NAMED_TUPLE_TYPES

include "mlir/Dialect/NamedTuple/IR/NamedTupleDialect.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/Traits.td"

//===----------------------------------------------------------------------===//
// NamedTupletype definitions
//===----------------------------------------------------------------------===//

class NamedTuple_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<NamedTuple_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def NamedTuple_NamedTupleType : NamedTuple_Type<"NamedTuple", "named_tuple", [
    MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface>
  ]> {
  let summary = "Fixed-sized collection of other types with a name";
  let description = [{
  }];
  let parameters = (ins "StringAttr":$name, Builtin_ArrayAttr:$member_types);
  let assemblyFormat = "`<` $name `,` $member_types `>`";
  let builders = [
    TypeBuilder<(ins "StringRef":$name, "ArrayRef<Type>":$types), [{
      llvm::SmallVector<mlir::Attribute> typeAttrs;
      for (auto &t : types)
        typeAttrs.push_back(mlir::TypeAttr::get(t));
      return $_get($_ctxt, mlir::StringAttr::get($_ctxt, name),
                   mlir::ArrayAttr::get($_ctxt, typeAttrs));
    }]>,
    // Empty tuple
    TypeBuilder<(ins "StringRef":$name), [{
      return $_get($_ctxt, mlir::StringAttr::get($_ctxt, name),
                   mlir::ArrayAttr::get($_ctxt,
                                        std::array<mlir::Attribute, 0>{}));
    }]>
  ];

  // \todo
  // let extraClassDeclaration = [{
  //   /// Return the number of held types.
  //   size_t size() const;

  //   /// Iterate over the held elements.
  //   using iterator = ArrayRef<Type>::iterator;
  //   iterator begin() const { return getTypes().begin(); }
  //   iterator end() const { return getTypes().end(); }

  //   /// Return the element type at index 'index'.
  //   Type getType(size_t index) const {
  //     assert(index < size() && "invalid index for tuple type");
  //     return getTypes()[index];
  //   }
  // }];
}

#endif // NAMED_TUPLE_TYPES
