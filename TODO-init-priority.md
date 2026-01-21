# TODO: Port init_priority Attribute Support

## Overview

Port support for the `init_priority` attribute on global variables, which controls the order of C++ global variable initialization.

## Current State

- Test `init_priority.cpp` fails with incorrect priority (uses 65535 instead of 101)
- Current output: `cir.global_ctors = [#cir.global_ctor<"_GLOBAL__sub_I_init_priority.cpp", 65535>]`
- Expected: `cir.global_ctors = [#cir.global_ctor<"__cxx_global_var_init", 101>]`

## Problem

Upstream is missing the check for `InitPriorityAttr` in LoweringPrepare.

## Incubator Solution

From `incubator/main:clang/lib/CIR/Dialect/Transforms/LoweringPrepare.cpp:911-913`:
```cpp
auto astDecl = mlir::cast<ASTDeclInterface>(*op.getAst());
if (astDecl.hasInitPriorityAttr())
  f.setGlobalCtorPriority(astDecl.getInitPriorityAttr()->getPriority());
```

## Options

### Option 1: Port AST Interface Infrastructure
- Port `ASTAttrInterfaces.td` from incubator
- This provides `ASTDeclInterface` with `hasInitPriorityAttr()` and `getInitPriorityAttr()`
- More complete solution, enables other AST attribute access

### Option 2: Add init_priority Attribute Directly to GlobalOp
- Add `OptionalAttr<I32Attr>:$init_priority` to GlobalOp in CIROps.td
- Set it during CodeGen when emitting global variables
- Simpler but less general

## Files to Modify

### Option 1:
- Add `clang/include/clang/CIR/Interfaces/ASTAttrInterfaces.td`
- Update `clang/lib/CIR/Dialect/Transforms/LoweringPrepare.cpp`

### Option 2:
- `clang/include/clang/CIR/Dialect/IR/CIROps.td` - Add init_priority to GlobalOp
- `clang/lib/CIR/CodeGen/CIRGenModule.cpp` - Set init_priority during emit
- `clang/lib/CIR/Dialect/Transforms/LoweringPrepare.cpp` - Read and use the priority

## Test

```bash
./build/bin/llvm-lit -v clang/test/CIR/IncubatorTests/CodeGen/init_priority.cpp
```
