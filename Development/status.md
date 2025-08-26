---
sort : 3
---
# Current Status

ClangIR can already be used with most common C sources and parts of C++.
C++ support is currently missing parts of cleanup and exception support,
which limits the amount of C++ code we can build and run - there's [on-going
work and improvement](https://github.com/llvm/clangir/issues) in the area though.

## Upstreaming

See [dedicated page](https://llvm.github.io/clangir/Development/upstreaming-progress.html).

## Code generation

### CIRGen
On language support:
- C/C++: C mostly covered, C++ has missing parts.
- Itanium ABI.
- Support for CUDA and OpenCL (both pass polibench).
- Missing Objective-C/C++ and anything not yet listed.

Targets:
- Linux/macOS x86_64
- Linux/macOS AArch64 (Android)
- SPIRV LLVM

To check those in details, see tests in
[CIR/CodeGen](https://github.com/llvm/clangir/tree/main/clang/test/CIR/CodeGen)
- they reflect supported features and targets.

### ABI Calling Convention lowering

Work-in-progress for both Linux x86_64 and AArch64, can be enabled with `-fclangir-call-conv-lowering`.

## Lifetime checker

Implementation of a C++ lifetime checker (for catching C++
dangling pointers) based in [C++ lifetime safety
paper](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1179r1.pdf).

Currently handles pointer/owner/aggregate type categories, lambda dangling
references via captures and can also catch idiomatic cases of dangling
references in coroutines `coawait` expressions.
