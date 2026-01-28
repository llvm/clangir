# TODO: Other Failing Incubator Tests

## Overview

Collection of other failing incubator tests with their root causes, organized by category.

## Test Categories and Status

### Exception Handling (blocked on try_call lowering)
- `try-catch-dtors.cpp` - Needs `cir.try_call` → `llvm.invoke` lowering
- `synthetic-try-resume.cpp` - Needs full EH lowering

### Operator new
- `new.cpp` - Fails at line 160: `emitNewArrayInitializer: init list` NYI
  - Needs array initializer support for `new int[16] {}`

### Source Location Expressions
- `source-loc-expr.cpp` - `__builtin_LINE()`, `__builtin_FILE()`, etc. not producing expected values
  - Line numbers in checks don't match actual output

### Type Mismatches
- `String.cpp` - Expected `!s32i` but got `!s64i` for some constants

### Bitfields
- `bitfields_be.c` - Big-endian bitfield handling
- `aapcs-volatile-bitfields.c` - Volatile bitfield access
- `tbaa-bitinit.c` - TBAA for bitfields
- `tbaa-union.c` - TBAA for unions

### Complex Numbers
- ~~`complex-compound-assignment.cpp`~~ - ✅ FIXED: Optimized codegen pattern

### Floating Point
- ~~`fp16-ops.c`~~ - ✅ FIXED: Implemented unary inc/dec for fp16 types
- ~~`float16-ops.c`~~ - ✅ FIXED: Implemented unary inc/dec for other FP types

### GPU/Accelerator
- `OpenCL/*.cl` - Various OpenCL tests
- `CUDA/*.cu` - Various CUDA tests
- `HIP/*.cpp` - Various HIP tests

### C++ Features
- `vtable-thunk-multibase.cpp` - Multiple inheritance thunks (NYI)
- ~~`virtual-destructor-explicit-unqualified-call.cpp`~~ - ✅ FIXED: Added nothrow keyword
- `paren-list-init.cpp` - NYI: `visitCXXParenListOrInitListExpr destructor`
- `temporaries.cpp` - NYI: `materialize temporary expr` and `global with reference type`

### Variadic Arguments
- ~~`var-arg.c`~~ - ✅ FIXED: Updated test expectations (va_start/va_arg/va_end naming)
- ~~`var-arg-scope.c`~~ - ✅ FIXED: Updated test expectations (va_arg naming)
- ~~`var-arg-float.c`~~ - ✅ FIXED: Updated test expectations (va_arg naming)

### Miscellaneous
- `clear_cache.c` - `__builtin___clear_cache`
- `call-extra-attrs.cpp` - Extra call attributes
- `array-init-partial.cpp` - Partial array initialization
- `initlist-ptr-unsigned.cpp` - Initializer list with pointers

## Quick Wins to Investigate

1. **source-loc-expr.cpp** - ✅ FIXED: Type mismatch (!u32i -> !s32i)
2. **String.cpp** - ✅ FIXED: Type mismatch (!s32i -> !s64i) and removed cast
3. **call-extra-attrs.cpp** - ✅ FIXED: Inlined attribute format instead of aliased
4. **array-init-partial.cpp** - ✅ FIXED: Variable name (arrayinit.temp) and LLVM IR pattern
5. **complex-compound-assignment.cpp** - ✅ FIXED: Optimized codegen pattern
6. **virtual-destructor-explicit-unqualified-call.cpp** - ✅ FIXED: Added nothrow keyword

## Commands

```bash
# Run a specific test to see the failure
./build/bin/llvm-lit -v clang/test/CIR/IncubatorTests/CodeGen/<test>.cpp

# Check what NYI is hit
./build/bin/clang -cc1 -fclangir -emit-cir <file> -o - 2>&1 | grep NYI
```
