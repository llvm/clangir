---
sort : 2
---

# Build and install

CIR is enabled by setting the cmake variable `CLANG_ENABLE_CIR` to `ON`, note
that it requires both `mlir` and `clang` projects to be enabled. Other
than that it works with a regular build of Clang/LLVM.

```
... -DLLVM_ENABLE_PROJECTS="clang;mlir;..." -DCLANG_ENABLE_CIR=ON...
```

See the [steps
here](https://llvm.org/docs/GettingStarted.html#local-llvm-configuration) for
general instruction on how to build LLVM.

For example, building and installing CIR enabled clang on macOS could look like:

```
CLANG=`xcrun -f clang`
INSTALLDIR=/tmp/install-llvm

$ cd llvm-project/llvm
$ mkdir build-release; cd build-release
$ /Applications/CMake.app/Contents/bin/cmake -GNinja \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=${INSTALLDIR} \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DLLVM_TARGETS_TO_BUILD="host" \
 -DLLVM_ENABLE_PROJECTS="clang;mlir" \
 -DCLANG_ENABLE_CIR=ON \
 -DCMAKE_CXX_COMPILER=${CLANG}++ \
 -DCMAKE_C_COMPILER=${CLANG} ../llvm
$ ninja install
```

Check for `cir-opt` to confirm all is fine:

```
$ /tmp/install-llvm/bin/cir-opt --help
```
