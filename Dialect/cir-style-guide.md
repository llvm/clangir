---
sort : 5
---

# CIR Assembly Style Guide

## Naming Convention

The name of a type, an attribute, or an operation should be in `snake_case` .
Examples:

```mlir
!cir.int
!cir.long_double<!cir.float>
!cir.data_member
#cir.cmp3way_info
#cir.dyn_cast_info
cir.binop
cir.get_member
```

The name of a type parameter, an attribute parameter, or an operation argument
and result, when defined in TableGen and printed in CIR assembly, should be in
`snake_case` .

Any keywords that appears in the syntax of a type, an attribute, or an
operation, should be in `snake_case` .

## Assembly Syntax Convention

### General Syntax Format

Each operation should follow the following general syntax:

```mlir
cir.op <args> <regions> `:` <typing> <attr>
```

Where:

- `<args>` is the operation’s arguments list.
- `<regions>` is any regions introduced by the operation.
- `<typing>` is the operation’s type specifications.
- `<attrs>` is any attributes attached to the operation.

### Arguments List

The arguments list should include all the arguments (except for some attributes
as specified in the “placement of attribute” section later) to the operation.
Don't use parenthesis around the arguments list. The most common format of an
arguments list is just a comma-separated list:

```mlir
%3 = cir.libc.memchr %0, %1, %2 : (!cir.ptr<!cir.void>, !s32i, !u64i) -> !cir.ptr<!cir.void>
```

Beyond that, one can also use auxiliary keywords and some mini-syntaxes in the
arguments list to make the operation’s assembly more readable:

```mlir
cir.store %0 to %1 : !s32i, !cir.ptr<!s32i>
%2 = cir.get_runtime_member %0[%1] : (!cir.ptr<!struct>, !cir.data_member<!s32i in !struct>) -> !cir.ptr<!s32i>
```

### Regions

For operations that introduces regions, the regions should follow the arguments
list in the assembly. One may use auxiliary keywords in the region list to make
the assembly more readable:

```mlir
cir.if %0 {
  ...
} else {
  ...
} : !cir.bool
```

```mlir
cir.ternary %2 true {
  ...
  cir.yield %0 : !s32i
} false {
  ...
  cir.yield %1 : !s32i
} : !cir.bool -> !s32i
```

### Type Specifications

The type specification gives the types of operation operands and results. Type
specifications are placed after the regions list (if any), separated from it
with a colon.

If the operation has trait `SameOperandsAndResultType` , the type specification
should only give a single type that represents the type of the operands and the
results:

```mlir
%1 = cir.cos %0 : !cir.float
%2 = cir.binop add %0, %1 : !s32i
```

Otherwise, the type specification can be further split into two type lists
separated by a right arrow ( `->` ). Before the arrow is a type list for the
operands, and after the arrow is a type list for the results. Types in a type
list are separated by commas; when a type list includes more than one type,
surround the type list with a parenthesis. If at least one of the two type
lists are empty, the empty type list(s) and the right arrow can be omitted.

If the operation has trait `SameTypeOperands` , the type list for the operands
should only has one type that give the type of the operands:

```mlir
%2 = cir.ptr_diff %0, %1 : !cir.ptr<!s32i> -> !u64i
```

Otherwise, the type specification should include a type for each of the
operands and the results. Examples:

```mlir
%2 = cir.get_runtime_member %0[%1] : (!cir.ptr<!struct>, !cir.data_member<!s32i in !struct>) -> !cir.ptr<!s32i>
```

### Placement of Attributes

Attributes on an operation can be classified into the following categories:

1. Attributes that affect the semantics of an operation. The operation becomes
   meaningless or changes semantics without the attribute. For example, a lot
   of operations take a `kind` attribute that specify which specific operation \
   to perform. These attributes are classified into this category.
2. Attributes that provide information for further CIR analysis,
   transformations, and translations, but do not affect the high-level
   semantics of the operation. AST reference attributes are typical examples in
   this category.
3. All other attributes that do not meet the conditions above.

For attributes in the first category, put them in the argument list. One may
place the attribute at any appropriate position within the argument list to
make the assembly readable. Example:

```mlir
%0 = cir.const #cir.int<42> : !s32i
%1 = cir.unary inc %0 : !s32i
%2 = cir.binop add %0, %1 : !s32i
%1 = cir.dyn_cast ptrcast %0 : !cir.ptr<!struct> -> !cir.ptr<!cir.void> #cir.dyn_cast_info
```

For the second category of attributes, put them in the `<attrs>` part before
any other attributes.

```mlir
%2 = cir.cmp3way %0, %1 : !s32i -> !u8i #cir.cmp3way<...>
```

One may use auxiliary keywords when specifying these attributes to make the
assembly more readable:

```mlir
%0 = cir.alloca !s32i init : !cir.ptr<!s32i> name("var")
```

For the third category of attributes, put them at the end of the `<attrs>` part
with the `attr-dict` TableGen directive.

## Unresolved Problems

TODO
