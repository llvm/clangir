---
parent: Getting started
nav_order: 3
---

# Testing

Tests are an important part on preventing regressions and covering new feature
functionality. There are multiple ways to run CIR tests.

## Unit tests

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

## [LLVM Test Suite](https://github.com/llvm/llvm-test-suite)

We often evaluate and monitor ClangIR support against singlesource, multisource and spec benchmarks. See the [benchmark results](https://llvm.github.io/clangir/Development/benchmark.html) page for more information.

### How to build and run

* Build a release version of ClangIR with assertions enabled (prevents some false positives).

* Cherry-pick [this git commit](https://github.com/bcardosolopes/llvm-test-suite/commit/87315cdd064aa2ba676575d3ec0e807cf84943c0) for extra CMake flags that enable ClangIR.

* Create a build directory and build the tests with the release version of ClangIR:
  ```bash
  cd <path-to>/test-suite
  rm -rf ./build && mkdir build && cd build
  cmake --no-warn-unused-cli ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=<path-to-clangir-build>/bin/clang \
    -DCMAKE_CXX_COMPILER=<path-to-clangir-build>/bin/clang++ \
    -C=../cmake/caches/O3.cmake \
    -DTEST_SUITE_SUBDIRS=SingleSource \ # MultiSource, External/SPEC/CINT2017rate, External/SPEC/CFP2017rate
    -DTEST_SUITE_CLANGIR_ENABLE=ON
  ```

* For SPEC [extra setup](https://llvm.org/docs/TestSuiteGuide.html#external-suites) is also needed.

* Build tests (`-k` ensures it won't stop if a test fails to compile):
  ```bash
  make -j -k
  ```

* In the build directory, run the tests with:
  ```bash
  lit -v .
  ```

### Generating a comparison Table

* Patch the `utils/compare.py` tool with:\
[002-generate-clangir-comparison-table.patch](../Files/002-generate-clangir-comparison-table.patch)

* Build the binaries and collect results for 15 runs of both baseline and ClangIR tests:
    ```bash
    # Set TEST_SUITE_CLANGIR_ENABLE=OFF to get the baseline tests.
    for i in {1..15}; do lit -v -o "baseline_$i.json" .; done;
    for i in {1..15}; do lit -v -o "clangir_$i.json" .; done;
    ```

* Create the comparison table using the patched `utils/compare.py`:
  ```bash
  utils/compare.py \
  --lhs-name baseline --rhs-name clangir --minimal-names \
  --merge-average --all -m compile_time -m exec_time \
  --csv-output results.csv \
  baseline_1.json ... baseline_15.json vs clangir_1.json ... clangir_15.json
  ```
