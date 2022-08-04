---
sort : 1
---

# Cloning repo

```
$ git clone https://github.com/llvm/clangir.git llvm-project
```
# Adding remote

Alternatively, one can just add remotes:

```
$ cd llvm-project

$ git remote add llvm-clangir git@github.com:llvm/clangir.git
$ git fetch llvm-clangir
$ git checkout -b clangir llvm-clangir/main
```
