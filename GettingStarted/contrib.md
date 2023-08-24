---
sort : 4
---

# How to contribute

Any change to the project should be done over github pull requests, anyone is welcome to contribute!

Code of conduct is the [same as LLVM](https://llvm.org/docs/CodeOfConduct.html).

## Monthly meetings

ClangIR is discussed monthly (on the fourth Monday) in the MLIR C/C++ Frontend Working Group, check the google calendar [here](https://calendar.google.com/calendar/u/0?cid=anZjZWFrbTNrYnBrdTNmNGpyc3YxbGtpZ29AZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ). The [living agenda document](https://docs.google.com/document/d/1iS0_4q7icTuVK6PPnH3D_9XmdcrgZq6Xv2171nS4Ztw) contain additional information on participation.

## Development Environment Setup

A few tips on how to set up a local ClangIR development environment using [VS Code](https://code.visualstudio.com/).

### Running ClangIR Docs Locally

ClangIR Docs (this website) is provided by [Github Pages](https://pages.github.com/). You can find its files in [ClangIR's repo `gh-pages` branch](https://github.com/llvm/clangir/tree/gh-pages).

* First, set up [VS Code's Dev Containers feature](https://code.visualstudio.com/docs/devcontainers/containers).
* Clone ClangIR and check out the `gh-pages` branch:
  ```base
  git clone --depth=1 -b gh-pages https://github.com/llvm/clangir.git cir-docs
  ```
* Open VS Code and go through the following steps:
	1. Press `ctrl + p` then run `>Dev Containers: Open Folder in Container`.
	1. Select ClangIR's repo checked out at `gh-pages`.
	1. Wait for VS Code to set up the development container.
	1. Access the dev container terminal with ```ctrl + ` ```.
	1. Run `bundle exec jekyll serve`.
* On your browser, open `http://127.0.0.1:4000/`.
* Edit the docs as necessary and update the page to see the changes.
