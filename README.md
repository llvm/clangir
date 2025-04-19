# ClangIR (CIR)

## Work-in-progress on CIR→MLIR lowering

This is a huge work-in-progress version adding new features used by [`aie++` C++
programming model](https://github.com/keryell/mlir-aie/tree/clangir), such as
better support for lowering through MLIR standard dialects.

What is experimented here:
- adding new features to CIR→MLIR lowering;
- cleaner lowering C/C++arrays as `tensor` for value semantics (for example in
  structs) and `memref` for reference semantics (all the other uses);
- fixing array support and allocation;
- allowing pointer to array or struct;
- adding support for class/struct/union by first experimenting with `tuple` and
  then introducing a new `named_tuple` dialect;
- implementation of more type of C/C++ casts;
- support struct member access through some `memref` flattening and casting
  casting;
- fixing a lot of verification bugs which are triggered when using ClangIR in a
  broader framework;
- enabling in-lining interface to CIR dialect;
- exposing some API to use ClangIR from a broader framework.

The output of this new lowering flow has not been tested yet followed by the
MLIR std→LLVMIR.

An alternative design could be to rely on some MLIR std→LLVMIR→MLIR std
back-and-forth traveling in the same module to have some parts of the LLVMIR
dialect to implement some missing features like it is done in the
[Polygeist](https://github.com/llvm/Polygeist) project.

## Main documentation

Check https://clangir.org for general information, build instructions and documentation.
