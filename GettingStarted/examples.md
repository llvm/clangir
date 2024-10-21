---
sort : 6
---

# Examples

## Enabling ClangIR

By default, Clang goes through its traditional CodeGen pipeline. After
you've [built a Clang-IR enabled Clang](build-install.html), you can
pass `-fclangir` to enable the ClangIR pipeline. [Godbolt](https://godbolt.org/z/9d8onnoT6)
also hosts a ClangIR-enabled compiler for quick experimentation.

## Emitting CIR

Pass `-Xclang -emit-cir` (in addition to `-fclangir`) to Clang to emit
CIR instead of assembly. [Godbolt](https://godbolt.org/z/hsEbzEGnY)
shows an example with various language features and how they're
translated to CIR.


## Viewing the pass pipeline

ClangIR runs its own pass pipeline. Some useful flags to introspect this
pipeline are:
* `-mmlir --mlir-print-ir-before-all` prints the CIR before each pass.
* `-mmlir --mlir-print-ir-after-all` prints the CIR after each pass.
* `-mmlir --mlir-print-ir-before=<pass>` and `-mmlir --mlir-print-ir-after=<pass>`
  print the CIR before and after a particular pass, respectively.

One particularly useful pass to introspect is `cir-lowering-prepare`
(LoweringPrepare), which goes from higher-level CIR constructs to
lower-level ones. Godbolt's [pipeline viewer](https://godbolt.org/z/1Ke8TKe7G)
is convenient for this.

## Generating LLVM IR

Pass `-fclangir -S -emit-llvm` to emit LLVM through the ClangIR
pipeline. [Godbolt](https://godbolt.org/z/KsGGWjEbq) shows an example.
All the standard Clang flags can be used as well, e.g. to
[build with optimizations](https://godbolt.org/z/4TvzrbnEn).

## Using the C++ lifetime-checker

### via clang driver
TBD

### via clang-tidy / clangd
TBD


