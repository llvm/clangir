#!/bin/bash
set -x -u -e

#python3.9 -m venv amd-venv
if [ -f amd-venv/bin/activate ]; then
  source amd-venv/bin/activate
fi
#export HTTP_PROXY=http://fwdproxy:8080
#export HTTPS_PROXY=http://fwdproxy:8080
#export NO_PROXY=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
#pip3 install click

PY=`echo $(uname) | if grep Linux >/dev/null; then echo python3.9; else echo /usr/bin/python3; fi`
echo "Python: $PY"

HOST_CC=/home/brunolopes/fbsource/fbcode/third-party-buck/platform010/build/llvm-fb/19/bin/clang

# Configure build if build directory doesn't exist
if [ ! -d "build" ]; then
  echo "Configuring ClangIR build..."
  cmake -S llvm -B build \
    -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=${HOST_CC} \
    -D CMAKE_CXX_COMPILER=${HOST_CC}++ \
    -D LLVM_ENABLE_PROJECTS="clang;mlir" \
    -D LLVM_ENABLE_RUNTIMES="compiler-rt" \
    -D CLANG_ENABLE_CIR=ON \
    -D LLVM_TARGETS_TO_BUILD='X86;AArch64' \
    -D LLVM_ENABLE_ASSERTIONS=ON \
    $@
fi

# Build and test ClangIR
ninja -C build clang llvm-profdata llvm-cov
