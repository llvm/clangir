# ClangIR TODO

## Crash Tests (8 XFAIL)

### Exception Handling (4 tests) - NYI
- `crashes/exception-handling-nyi.cpp` - Basic try/catch blocks, crashes at emitBeginCatch
- `crashes/copy-on-catch.cpp` - Exception copy-on-catch semantics
- `crashes/exception-ptr.cpp` - std::make_exception_ptr
- `crashes/async-future.cpp` - std::async/std::future (depends on exceptions)

**Missing:** Personality functions, landing pads, invoke instructions, exception object handling

### Static Initialization (1 test) - NYI
- `crashes/static-local-guard-nyi.cpp` - Thread-safe static local initialization

**Missing:** `__cxa_guard_acquire`/`__cxa_guard_release` runtime calls per Itanium C++ ABI

### GNU Extensions (1 test) - NYI
- `crashes/computed-goto-nyi.cpp` - Computed goto (&&label syntax)

**Missing:** AddrLabelExpr handling in constant expression emitter

### Array Allocation (1 test)
- `crashes/array-new-default-arg.cpp` - Array new with default constructor arguments

**Error:** Backend lowering failure, operation has uses after destruction

### Type System (1 test)
- `crashes/filesystem-sd-automatic.cpp` - Comparison operators emit wrong type

**Error:** `'cir.if' op operand #0 must be CIR bool type, but got '!cir.int<u, 1>'`

**Note:** Branch-through cleanup crash was fixed, this is remaining type checking issue

## Calling Convention / ABI Divergences (32 tests)

### Struct Coercion - x86_64 System V ABI (18 tests)

Small structs (≤16 bytes) should be coerced to integer types per ABI, CIR returns struct:
- `divergences/calling-conv-4byte-struct.cpp` - Should coerce to i32
- `divergences/calling-conv-12byte-struct.cpp` - Should coerce to {i64, i32}
- `divergences/calling-conv-16byte-struct.cpp` - Should coerce to {i64, i64}
- `divergences/calling-conv-20byte-struct.cpp` - Should use sret for large structs
- `divergences/calling-conv-longlong-struct.cpp` - Should coerce to i64
- `divergences/calling-conv-bool-in-struct.cpp` - Should coerce to i64
- `divergences/calling-conv-array-in-struct.cpp` - Should coerce to i64
- `divergences/calling-conv-bitfield-struct.cpp` - Bitfield packing divergence
- `divergences/calling-conv-empty-struct.cpp` - Empty struct (size 1 in C++)
- `divergences/calling-conv-aligned-struct.cpp` - Over-aligned structs
- `divergences/calling-conv-nested-struct.cpp` - Should coerce to {i64, i64}
- `divergences/calling-conv-packed-struct.cpp` - __attribute__((packed))
- `divergences/calling-conv-pointer-in-struct.cpp` - Pointer in struct
- `divergences/calling-conv-multiple-struct-params.cpp` - Multiple struct parameters
- `divergences/calling-conv-two-longlongs.cpp` - Two long longs
- `divergences/small-struct-coercion.cpp` - General ≤8 byte struct coercion
- `divergences/sret-abi-mismatch.cpp` - Non-trivial returns need sret (hidden first param)
- `divergences/array-new-delete-divergences.cpp` - Array new/delete attributes

**Impact:** Binary incompatibility with standard CodeGen
**CodeGen Reference:** `clang/lib/CodeGen/TargetInfo.cpp` (x86_64ABIInfo)

### Floating-Point Struct Coercion - x86_64 SSE (5 tests)

Structs with floats should use SSE class (XMM registers):
- `divergences/float-single-float-struct.cpp` - Single float, use XMM
- `divergences/float-double-struct.cpp` - Double, use XMM
- `divergences/float-two-floats-struct.cpp` - Two floats, use XMM
- `divergences/float-mixed-int-float.cpp` - Mixed INTEGER/SSE classification
- `divergences/float-struct-calling-conv.cpp` - General float struct handling

**Impact:** Binary incompatibility, incorrect register usage

### Member Pointer ABI (12 tests)

Member function pointers passed as {i64, i64} aggregate instead of decomposed scalars:
- `divergences/member-ptr-abi-calling-conv.cpp` - Should decompose to two i64 params
- `divergences/member-ptr-data-member.cpp` - Data member pointer
- `divergences/member-ptr-virtual-function.cpp` - Virtual function pointer
- `divergences/member-ptr-multiple-inheritance.cpp` - With multiple inheritance
- `divergences/member-ptr-overloaded-function.cpp` - Overloaded function
- `divergences/member-ptr-const-method.cpp` - Const method
- `divergences/member-ptr-null.cpp` - Null member pointer
- `divergences/member-ptr-comparison.cpp` - Comparison operators
- `divergences/member-ptr-base-to-derived.cpp` - Base-to-derived conversion
- `divergences/member-ptr-stored-in-struct.cpp` - As struct member
- `divergences/member-ptr-returning-struct.cpp` - Returning in struct
- `divergences/member-ptr-array.cpp` - Array of member pointers

**Impact:** Binary incompatibility per System V x86_64 ABI

## Missing Comdat Groups (31 tests)

Templates, constructors, destructors, lambdas, and other inline definitions missing `comdat` attribute on `linkonce_odr` symbols, causing ODR violations and code bloat.

### Templates (8 tests)
- `divergences/template-missing-comdat.cpp`
- `divergences/template-class-instantiation.cpp`
- `divergences/template-member-function.cpp`
- `divergences/template-multiple-type-params.cpp`
- `divergences/template-non-type-param.cpp`
- `divergences/template-specialization.cpp`
- `divergences/template-variadic.cpp`
- `divergences/template-inheritance.cpp`

### Constructors/Destructors (9 tests)
- `divergences/inline-ctor-dtor-missing-comdat.cpp` - C1, C2, D1, D2 variants
- `divergences/ctor-parameterized.cpp`
- `divergences/ctor-copy.cpp`
- `divergences/ctor-move.cpp`
- `divergences/ctor-delegating.cpp`
- `divergences/ctor-member-init-list.cpp`
- `divergences/ctor-inherited.cpp`
- `divergences/ctor-multiple-inheritance.cpp`
- `divergences/ctor-deep-inheritance.cpp`

### Lambdas (7 tests)
- `divergences/lambda-missing-comdat.cpp` - operator() comdat
- `divergences/lambda-simple.cpp`
- `divergences/lambda-with-params.cpp`
- `divergences/lambda-capture-by-value.cpp`
- `divergences/lambda-capture-by-ref.cpp`
- `divergences/lambda-mutable.cpp`
- `divergences/lambda-returning-struct.cpp`

### Inheritance (5 tests)
- `divergences/inheritance-missing-comdat.cpp`
- `divergences/inheritance-diamond.cpp`
- `divergences/inheritance-empty-base.cpp`
- `divergences/inheritance-private.cpp`
- `divergences/inheritance-protected.cpp`

### Other (2 tests)
- `divergences/vtable-missing-comdat.cpp` - VTables and RTTI (_ZTV*, _ZTI*, _ZTS*)
- `divergences/operator-missing-comdat.cpp` - Operator overloads

**Impact:** ODR violations in multi-TU programs, linker errors, code bloat

## RTTI and Virtual Inheritance (4 tests)

- `divergences/rtti-linkage-gep.cpp` - Type info missing linkonce_odr/comdat, GEP type mismatch (byte vs ptr based)
- `divergences/rtti-dynamic-cast-upcast.cpp` - Dynamic cast upcast
- `divergences/rtti-dynamic-cast-downcast.cpp` - Dynamic cast downcast
- `divergences/virtual-inheritance-vtt.cpp` - VTT missing comdat, type info linkage, unnamed_addr, inrange annotations, string null terminators

**Impact:** ODR violations, linking issues

## Missing LLVM Attributes and Metadata (2 tests)

- `divergences/missing-llvm-attributes.cpp` - Missing:
  - Parameter attributes: noundef, nonnull, dereferenceable(N)
  - Function attributes: mustprogress, unnamed_addr
  - Metadata: min-legal-vector-width, target-features, stack-protector-buffer-size
- `divergences/thread-local-wrapper-missing.cpp` - Missing `_ZTW*` wrapper functions for thread_local variables per Itanium C++ ABI

**Impact:** Reduced optimization quality, missed UB detection, TLS initialization issues

## Static and Global Variables (3 tests)

- `divergences/static-local-trivial.cpp` - Static local with trivial initialization
- `divergences/static-member-variable.cpp` - Static member variable
- `divergences/static-inline-member.cpp` - Static inline member variable
- `divergences/global-constructor.cpp` - Global object with constructor, @llvm.global_ctors

**Impact:** Various initialization and linkage issues

## Code Quality (1 test)

- `divergences/unnecessary-temp-allocas.cpp` - Unnecessary temporary allocas for return values

**Impact:** Verbose IR (likely optimized away by LLVM but unnecessary)

## Test Statistics

| Category | Total | XFAIL | Pass | Pass Rate |
|----------|-------|-------|------|-----------|
| Crash Tests | 10 | 8 | 2 | 20% |
| Divergence Tests | 77 | 77 | 0 | 0% |
| **Total** | **87** | **85** | **2** | **2.3%** |

### By Issue Type
| Issue | Tests | Priority |
|-------|-------|----------|
| Exception Handling | 4 | 🔴 Critical |
| Static Local Guards | 1 | 🔴 High |
| Type System | 1 | 🔴 High |
| Struct Coercion ABI | 18 | 🟠 High (ABI) |
| Float Coercion ABI | 5 | 🟠 High (ABI) |
| Member Ptr ABI | 12 | 🟠 High (ABI) |
| Missing Comdat | 31 | 🟡 Medium (ODR) |
| RTTI/Virtual | 4 | 🟡 Medium |
| Computed Goto | 1 | 🟡 Medium |
| Array New/Delete | 1 | 🟡 Medium |
| Attributes/Metadata | 2 | 🟢 Low |
| Static/Global Vars | 3 | 🟢 Low |
| Code Quality | 1 | 🟢 Low |
