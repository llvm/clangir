---
parent: Development
nav_order: 6
---

## Compile time

A report on on build time investigations around CIR has been presented here:
- Jun 2025: Asia LLVM. *ClangIR’s Footprint: A quick compile-time impact report*. [video](https://www.youtube.com/watch?v=Dh_RObp5SUE), [pdf](Files/asiallvm-brunolopes-25-cir-compile-time.pdf)

In a gist, a total of 57 sources from `Multisource` benchmark take above 0.5 seconds and above to compile under `-O2`, and the slowdown breakdown is:
  - 30 sources found in 20-30% slowdown
  - 20 sources found in 10-20% slowdown
  - 7 sources found in 30% and above slowdown.

Some edge cases, like `clamr_cpuonly.cpp` (53% slowdown under `-O0`),  were [triaged](https://github.com/llvm/clangir/issues/1865) and found to be a low hanging fruits for compile time improvements. Since we still miss support for some C++ features, we haven't been able to evaluate more C++ code.
