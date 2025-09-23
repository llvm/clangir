---
parent: Getting started
nav_order: 1
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

# Working across rebases

The ClangIR fork’s `main` branch is regularly rebased on top of LLVM’s
main branch, to keep up with the latest upstream development. Rebases
are announced on the [Discord channel](https://discord.com/channels/636084430946959380/1034236948421804074).
Whenever a rebase is pushed, you’ll need to update your local branches
to avoid rebase conflicts. If you don’t already have a preferred git
workflow for this, our recommended one is:

* If you have a branch with no local commits, run the following to
  update it:
```
git fetch origin main
git reset --hard origin/main
```

* If you have a branch with local commits, count the number of local
  commits you have, which we’ll call N, and then run:
```
git fetch origin main
git rebase HEAD~<N> --onto origin/main
```
