---
sort : 2
---

# Build and install

In order to enable CIR related functionality, just add `mlir`
and `clang` to the CMake list of *enabled projects* and do a regular
LLVM build.

```
... -DLLVM_ENABLE_PROJECTS="clang;mlir;..." ...
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
 -DLLVM_TARGETS_TO_BUILD="X86" \
 -DLLVM_ENABLE_PROJECTS="clang;mlir" \
 -DCMAKE_CXX_COMPILER=${CLANG}++ \
 -DCMAKE_C_COMPILER=${CLANG} ../
$ ninja install
```

Check for `cir-tool` to confirm all is fine:

```
$ /tmp/install-llvm/bin/cir-tool --help
```
