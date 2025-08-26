---
sort : 2
---
# ClangIR in practice

ClangIR is an MLIR dialect tightly integrated with Clang, meaning it resides within the Clang project rather than as a standalone MLIR in-tree dialect. Some CIR operations optionally include backreferences to the Clang AST, allowing analysis and transformation passes to leverage AST information when needed. This design also supports progressive lowering by enabling late use of AST nodes.

When you pass the flag `-fclangir` to the Clang driver, the compilation pipeline is adjusted so that CIR is emitted from Clang's AST and then lowered through LLVM IR and the backend stages. To print CIR during compilation, you can use the `-emit-cir` flag, which stops the compilation after the last
CIR pass (use `-fclangir-disable-passes` to stop right after the AST translation into CIR, see more about `CIRGen` below).

The diagram below (last update: Jan 2024) illustrates how the compiler pipeline operates:

![](/../Images/2024-Jan-Pipeline.png)

Some terms often used in CIR or as part of the CIR pipeline:

- CIRGen: the process of generating CIR out of the Clang AST, before any CIR based
pass is executed in the pipeline.
- LLVM lowering: generating LLVM IR out of CIR.
- Through-MLIR lowering: Lower CIR to several MLIR core dialects.

For information on how to build Clang with CIR support see
[instructions](https://llvm.github.io/clangir/GettingStarted/build-install.html).
Check out [examples](https://llvm.github.io/clangir/GettingStarted/examples.html)
of CIR usage and capabilities.
