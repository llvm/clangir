---
sort : 1
---
# Project Motivation

In short, an IR that captures higher-level C/C++ semantics enables a range of idiomatic diagnostics and performance optimizations that are difficult to achieve using either the Clang AST or LLVM IR.

ClangIR uses MLIR as its compiler framework, enabling the development of passes and IR, rapid design iteration, and reuse of community-provided analyses and transformations that can be easily adapted for ClangIR.

# History

ClangIR [started](https://github.com/facebookarchive/clangir) in October 2021, at Meta. In June 2022 an [RFC was posted](https://discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319) proposing to incorporate the project (still in very early stages) into the llvm-project. The outcome of this RFC lead to ClangIR becoming part of the LLVM incubator, where the development has been happening since then. In January 2024, as a graduation step from incubator, the [upstreaming RFC](https://discourse.llvm.org/t/rfc-upstreaming-clangir/76587) kicked the discussion about upstreaming again. The RFC was accepted and there's a high influx of upstreaming happening (as of Aug 2025), see [current status](https://llvm.github.io/clangir/Development/benchmark.html) for more progress information.
