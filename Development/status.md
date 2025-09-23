---
parent: Development
nav_order: 3
---
# Current Status

ClangIR already supports most common C source code and some parts of C++. However, C++ support is still missing certain cleanup and exception handling features, which restricts the range of C++ code that can be built and executed. There is ongoing [work and improvements](https://github.com/llvm/clangir/issues) addressing these limitations.

Check out the current supported set of [benchmarks](https://llvm.github.io/clangir/Development/benchmark.html).

## Upstreaming

See [Upstreaming Progress](https://llvm.github.io/clangir/Development/upstreaming-progress.html).

## Code generation

### CIRGen
Language support:
- C/C++: C mostly covered, C++ has missing parts.
- Itanium ABI.
- Support for CUDA and OpenCL (both pass polibench).
- Missing Objective-C/C++ and anything not yet listed.

Targets:
- Linux/macOS x86_64
- Linux/macOS AArch64 (Android)
- SPIRV LLVM

To check those in details, see tests in [CIR/CodeGen](https://github.com/llvm/clangir/tree/main/clang/test/CIR/CodeGen), they reflect supported features and targets.

### ABI Calling Convention lowering

Work-in-progress for both Linux x86_64 and AArch64, can be enabled with `-fclangir-call-conv-lowering`.

## Lifetime checker

Implementation of a C++ lifetime checker (for catching C++
dangling pointers) based in [C++ lifetime safety
paper](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1179r1.pdf).

Currently handles pointer/owner/aggregate type categories, lambda dangling
references via captures and can also catch idiomatic cases of dangling
references in coroutines `coawait` expressions.
