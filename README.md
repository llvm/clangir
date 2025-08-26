# ClangIR (CIR)

ClangIR is a high-level representation in Clang that reflects aspects of the C/C++
languages and their extensions. It is implemented using MLIR and occupies a position
between Clang's AST and LLVM IR.

The project is [incubated](https://github.com/llvm/clangir) under LLVM's GitHub
umbrella and is being [upstreamed](https://github.com/llvm/llvm-project/labels/ClangIR)
to the llvm-project (LLVM's official main repository).

[Follow the progress track](https://llvm.github.io/clangir/Development/benchmark.html)
for updates.

# In the web

- Oct 2023: *Evolution of ClangIR: A Year of Progress, Challenges, and Future Plans*. US LLVM Developers Meeting. [video](https://www.youtube.com/watch?v=XNOPO3ogdfQ), [pdf](http://brunocardoso.cc/resources/2023-LLVMDevMtgClangIR.pdf).
- June 2022: [RFC: An MLIR based Clang IR (CIR)](https://discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319)

# Where to go from here?

Check out our docs for [contributing to the
project](https://llvm.github.io/clangir/GettingStarted/contrib.html) and get a
tour in the [CIR Dialect](https://llvm.github.io/clangir/Dialect/), or the list
of [passes](https://llvm.github.io/clangir/Dialect/passes.html) written on
top of CIR and are part of the CIR pipeline in Clang.

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

*Last updated: Tue Jul  1 17:08:48 PDT 2025*
