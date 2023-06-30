---
sort : 3
---

# Testing

Tests are an important part on preventing regressions and covering new feature
functionality. There are multiple ways to run CIR tests.

The more aggresive (slower) one:
```
$ ninja check-all
```

CIR specific test targets using ninja:
```
$ ninja check-clang-cir
$ ninja check-clang-cir-codegen
```

Using `lit` from build directory:

```
$ cd build-release
$ ./bin/llvm-lit -a ../../clang/test/CIR
```
