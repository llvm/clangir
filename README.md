# Clang IR (CIR)

Clang IR (CIR) is a new IR for Clang. ClangIR (CIR) is built on top of MLIR and
it's basically a mlir dialect for C/C++ based languages in Clang.  It's
representation level sits somewhere between Clang AST and LLVM IR.

# Motivation

In a gist, an IR that can cover C/C++ higher level semantics enables a class of
idiomatic diagnostics and performance optimizations that could be currently hard to
explore on Clang AST or LLVM IR level.

By using MLIR, ClangIR leverages on a compiler framework to write passes, IR and quickly
iterate design, while re-using community provided analysis and transformations that can
be easily adapted for CIR.

The [LLVM's discourse
RFC](https://discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319) goes in
depth about the initial project motivation, status and design choices.

# What's ClangIR in practice?

ClangIR is a MLIR dialect that is also bound to Clang, meaning it lives inside
the clang project and not as a mlir in-tree dialect. Some CIR operations
optionally contain backreferences to the Clang AST, enabling analysis and
transformation passes to optionally use AST information, while also allowing
progressive lowering through late use of AST nodes.

By passing `-Xclang -fclangir-enable` to the clang driver, the compilation
pipeline is modified and CIR gets emitted from ClangAST and then lowered to
LLVM IR, backend, etc. Since our LLVM emission support is WIP, functionality is
currently limited. To get CIR printed out of a compiler invocation, the flag
`-Xclang -emit-cir` can be used, which will force the compiler to stop right
after CIR is produced.

Instructions on how to build clang with ClangIR support can be found
[here](https://llvm.github.io/clangir/GettingStarted/build-install.html).

# Examples

## Emitting CIR
TBD

## Generating LLVM IR
TBD

## Using the C++ lifetime-checker via clang driver
TBD

## Using the C++ lifetime-checker via clang-tidy
TBD

# Current status

The project is active, here's a list of the current supported pieces:

- CIRGen: the process of generating CIR out of the Clang AST. We support a good
set of functionality from C/C++, but there are many missing. The
[CIR/CodeGen](https://github.com/llvm/clangir/tree/main/clang/test/CIR/CodeGen)
test directory is a good proxy of the current supported features.

- LLVM lowering: generating LLVM IR out of CIR. About 50% of all programs in
`llvm-testsuite/SingleSource` pass correctness checks.

- MLIR in-tree dialect lowering: basically CIR -> MLIR dialects, initial
support to memref and some other dialects but currently not as active as LLVM
lowering.

- Lifetime checker: implementation of a C++ lifetime checker (for catching C++
dangling pointers) based in [C++ lifetime safety
paper](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p1179r1.pdf).
Currently handles pointer/owner/aggregate type categories, lambda dangling
references via captures and can also catch idiomatic cases of dangling
references in coroutines `coawait` expressions.

# Where to go from here?

Check out our docs for [contributing to the
project](https://llvm.github.io/clangir/GettingStarted/contrib.html) and get a
tour in the [CIR Dialect](https://llvm.github.io/clangir/Dialect/).

# Inspiration

ClangIR is inspired in the success of other languages that greatly benefit from
a middle-level IR, such as
[Swift](https://apple-swift.readthedocs.io/en/latest/SIL.html) and
[Rust](https://rustc-dev-guide.rust-lang.org/mir/index.html). Particularly,
optionally attaching AST nodes to CIR operations is inspired by SIL references
to AST nodes in Swift.

# Github project

This project is part of the LLVM incubator, the source of truth for CIR is found at
[https://github.com/llvm/clangir](https://github.com/llvm/clangir).

The [main](https://github.com/facebookincubator/clangir/tree/main) branch
contains a stack of commits, occasionally rebased on top of LLVM upstream,
tracked in
[base](https://github.com/llvm/clangir/tree/base)
branch.

<!---
On vim use ":r!date"
-->
*Last updated: Fri Aug 11 15:32:01 PDT 2023*
