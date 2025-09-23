---
parent: Getting started
nav_order: 5
---

# Coding Guideline

This document describes two different aspects of coding for ClangIR: style and project best practices.

## Style

CIR coding style can abide to LLVM or MLIR, depending on the parts of the project being touched. This short doc gives some guidelines.

See existing coding styles here:
- [LLVM Coding Standard](https://llvm.org/docs/CodingStandards.html)
- [MLIR Style Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide)

This document is likely to change as the project evolves, the last relevant discussion happened [here](https://github.com/llvm/llvm-project/pull/91007).

### Use of `auto`

1. Use it for function templates such as `isa`, `dyn_cast`, `cast`, `create` and `rewriteOp.*` variants.
2. Use best judgement for anywhere the type might be easier to infer from the context.

Other usages of `auto` that are more common and idiomatic in MLIR, need to be avoided in parts where the context is more related to Clang, helping Clang reviewers to be more efficient. This means limiting to the above rules (as encoded LLVM Coding Standard) in `lib/CIR/{CodeGenFrontendAction}` and other directories outside `lib/CIR`.

However, for `lib/CIR/{Transforms,*}`, it should be fine to be more idiomatic, e.g.
- Operation queries: `DepthwiseConv2DNhwcHwcmOp op; ...; auto stride = op.getStrides();`.
- MLIR locations: `auto loc = op->getLoc()` is usually done everywhere since MLIR always require `mlir::Location`s when creating operations.

Note that if the reviewer would still prefer `auto` not to be used, one should follow the request for change.

### Variable naming: CamelCase vs camelBack

Method and class names should just follow LLVM, but for the variable names one should follow the code structure:

- `clang/include` -> LLVM
- `lib/Driver/*` -> LLVM
- `lib/FrontendTool` -> LLVM
- `lib/CIR/FrontendAction` -> LLVM
- `lib/CIR/CodeGen` -> MLIR
- `lib/CIR/*` -> MLIR

Sometimes it might be tricky to decide, in which case think about these two aspects:
- The convention should be consistent with other nearby code, whatever that convention happens to be.
- Avoid busywork but use your best judgement on how we eventually get to a consistent use of the LLVM style. If changing to LLVM style means touching 1000s of lines of code while changing to MLIR style means touching 10s of lines of code, switch to MLIR style. But if changing to LLVM style means touching 100 lines of code and changing to MLIR style means touching 150 lines of code, it's probably best to just bite the bullet and switch to LLVM style even though MLIR would be less effort.
- If unsure, be consistent but avoid mixing styles. Example:
```
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const clang::CodeGenOptions &CGO,
               clang::DiagnosticsEngine &Diags);
```
should be:
```
  CIRGenModule(mlir::MLIRContext &Context, clang::ASTContext &AstCtx,
               const clang::CodeGenOptions &CGO,
               clang::DiagnosticsEngine &Diags);
```
or
```
  CIRGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const clang::CodeGenOptions &cgo
               clang::DiagnosticsEngine &diags);
```


## Best practices

### Traditional codegen skeleton
TBD

### Unrecheable and assertions
TBD

### Tests
TBD

