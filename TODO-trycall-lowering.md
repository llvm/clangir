# TODO: Implement cir.try_call to LLVM invoke Lowering

## Overview

Port the exception handling lowering from incubator to enable `cir.try_call` → `llvm.invoke` conversion.

## Current State

- `cir.try_call` is generated during CodeGen and flattened correctly
- LLVM lowering fails with: `failed to legalize operation 'cir.try_call'`
- Upstream already has: `CIRToLLVMEhInflightOpLowering`, `CIRToLLVMResumeFlatOpLowering`

## Missing Lowering Patterns

### 1. CIRToLLVMTryCallOpLowering

Converts `cir.try_call` to `llvm.invoke`.

From incubator (`LowerToLLVM.cpp:1638-1648`):
```cpp
mlir::LogicalResult CIRToLLVMTryCallOpLowering::matchAndRewrite(
    cir::TryCallOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (op.getCallingConv() != cir::CallingConv::C) {
    return op.emitError(
        "non-C calling convention is not implemented for try_call");
  }
  return rewriteToCallOrInvoke(op.getOperation(), adaptor.getOperands(),
                               rewriter, getTypeConverter(), op.getCalleeAttr(),
                               op.getCont(), op.getLandingPad());
}
```

### 2. Update rewriteCallOrInvoke

Add `continueBlock` and `landingPadBlock` parameters. When `landingPadBlock` is provided, create `mlir::LLVM::InvokeOp` instead of `CallOp`:

```cpp
if (landingPadBlock) {
  auto newOp = rewriter.replaceOpWithNewOp<mlir::LLVM::InvokeOp>(
      op, llvmFnTy, calleeAttr, callOperands, continueBlock,
      mlir::ValueRange{}, landingPadBlock, mlir::ValueRange{});
  // set calling conv, etc.
} else {
  // existing CallOp creation
}
```

### 3. CIRToLLVMCatchParamOpLowering

Converts `cir.catch_param begin/end` to `__cxa_begin_catch`/`__cxa_end_catch` calls.

From incubator (`LowerToLLVM.cpp:4552-4578`).

## Files to Modify

1. `clang/lib/CIR/Lowering/DirectToLLVM/LowerToLLVM.cpp`
   - Add `CIRToLLVMTryCallOpLowering` pattern
   - Add `CIRToLLVMCatchParamOpLowering` pattern
   - Update `rewriteCallOrInvoke` to handle invoke generation
   - Register new patterns in `populateCIRToLLVMConversionPatterns`

2. `clang/lib/CIR/Lowering/DirectToLLVM/LowerToLLVM.h`
   - Declare new lowering pattern classes (if not auto-generated)

## Tests to Verify

```bash
# Simple destructor test
./build/bin/clang -cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
  -fcxx-exceptions -fexceptions -fclangir -emit-llvm /tmp/test-dtor.cpp -o -

# Incubator tests
./build/bin/llvm-lit -v clang/test/CIR/IncubatorTests/CodeGen/try-catch-dtors.cpp
./build/bin/llvm-lit -v clang/test/CIR/IncubatorTests/Lowering/try-catch.cpp
```

## Reference Commits in Incubator

- Look at incubator/main for `CIRToLLVMTryCallOpLowering` implementation
- Search: `git show incubator/main:clang/lib/CIR/Lowering/DirectToLLVM/LowerToLLVM.cpp`
