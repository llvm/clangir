---
sort : 1
---
# Project Motivation

In short, an IR that captures higher-level C/C++ semantics enables a range of idiomatic diagnostics and performance optimizations that are difficult to achieve using either the Clang AST or LLVM IR.

ClangIR uses MLIR as its compiler framework, enabling the development of passes and IR, rapid design iteration, and reuse of community-provided analyses and transformations that can be easily adapted for ClangIR.
