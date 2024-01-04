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

## External

Currently we do not evaluate ClangIR against external test suites automatically, this, however, can be achieved with some manual work.

### [LLVM Test Suite](https://github.com/llvm/llvm-test-suite)

So far, we have tested ClangIR **only against the SingleSource** tests from this suite.

A table detailing ClangIR's status for each tests can be found [here](../Standalone/single-source-tests-table.md).

Currently, 51% (935/1824) of the SingleSource tests are passing. A good way to start contributing to ClangIR is to pick one of the `NOEXE` of `FAIL` tests and try to patch it!

#### How to Run

* Build a release version of ClangIR with assertions enabled (prevents some false positives).
* Apply the following patch to create a CMake flag that will enable ClangIR: \
  [001-enable-clangir-on-singlesouce.patch](../Files/001-enable-clangir-on-singlesouce.patch)

* Create a build directory and build the tests with the release version of ClangIR:
  ```bash
  cd <path-to>/test-suite
  rm -rf ./build && mkdir build && cd build
  cmake --no-warn-unused-cli ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=<path-to-clangir-build>/bin/clang \
    -DCMAKE_CXX_COMPILER=<path-to-clangir-build>/bin/clang++ \
    -C=../cmake/caches/O3.cmake \
    -DTEST_SUITE_SUBDIRS=SingleSource \
    -DTEST_SUITE_CLANGIR_ENABLE=ON
  ```

* Build tests (`-k` ensures it won't stop if a test fails to compile):
  ```bash
  make -j -k
  ```

* In the build directory, run the tests with:
  ```bash
  lit --timeout=60 -v .
  ```

#### Generating the Comparison Table

* Patch the `utils/compare.py` tool with:\
[002-generate-clangir-comparison-table.patch](../Files/002-generate-clangir-comparison-table.patch)

* Build the binaries and collect results for 15 runs of both baseline and ClangIR tests:
    ```bash
    # Set TEST_SUITE_CLANGIR_ENABLE=OFF to get the baseline tests.
    for i in {1..15}; do lit --timeout=60 -v -o "baseline_$i.json" .; done;
    for i in {1..15}; do lit --timeout=60 -v -o "clangir_$i.json" .; done;
    ```

* Create the comparison table using the patched `utils/compare.py`:
  ```bash
  utils/compare.py \
  --lhs-name baseline --rhs-name clangir --minimal-names \
  --merge-average --all -m compile_time -m exec_time \
  --csv-output results.csv \
  baseline_1.json ... baseline_15.json vs clangir_1.json ... clangir_15.json
  ```
