---
sort : 4
---

# How to contribute

Any change to the project should be done over github pull requests, anyone is welcome to contribute!

Code of conduct is the [same as LLVM](https://llvm.org/docs/CodeOfConduct.html).

## Community

### Monthly meetings

ClangIR is discussed monthly (on the fourth Monday) in the MLIR C/C++ Frontend Working Group, check the google calendar [here](https://calendar.google.com/calendar/u/0?cid=anZjZWFrbTNrYnBrdTNmNGpyc3YxbGtpZ29AZ3JvdXAuY2FsZW5kYXIuZ29vZ2xlLmNvbQ). The [living agenda document](https://docs.google.com/document/d/1-flHK3TjQUrkSO2Fdt4webZ2zCyeXxpTLMiRQbMW7hE) contain additional information on participation.

### Have a question?

Find us on [LLVM discord](https://discord.gg/xS7Z362), under the `#clangir`
channel. You can also [create a new github
issue](https://github.com/llvm/clangir/issues) to start a discussion.

## Development Environment Setup

A few tips on how to set up a local ClangIR development environment.

### VS Code Setup <sub><small style="font-weight: normal;">(Tested on Ubuntu jammy)</small></sub>

Start by forking ClangIR's repository then cloning your fork to your machine.

#### CMake Setup

Install the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) plugin and the following dependencies:
```bash
sudo apt install clang lld ccache cmake ninja-build
```

In `.vscode/settings.json`, add the following configurations:
```json
"cmake.sourceDirectory": "${workspaceFolder}/llvm",
"cmake.buildDirectory": "${workspaceFolder}/build/${buildType}",
"cmake.generator": "Ninja",
"cmake.parallelJobs": 8, // Adjust to your machine's thread count.
"cmake.configureOnOpen": true,
```

Copy the [cmake-variants.json](../Files/cmake-variants.json) file from this repository into your `.vscode` folder. These configurations aim to reduce compile time for faster incremental development (change, recompile, and test).

On VS Code's bottom bar, select the `Debug` variant, your installed Clang version, and `[all]` as the build's target. Finally, click `Build` and wait for it to build.

![](../Images/cmake-integration-build.png)

#### Clangd Setup

Install clangd (`sudo apt install clangd`) and its [VS Code plugin](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd).

In `.vscode/settings.json` add the following configurations:
```json
"clangd.arguments": [
    "--background-index",
    "--compile-commands-dir=${workspaceFolder}/build/Debug/"
],
```

Open any `.cpp` file in the project to trigger clangd indexing.

#### Useful Tasks

In this [tasks.json](../Files/tasks.json) there are several VS Code tasks useful for development. Copy this file into your `.vscode` folder to use it. These tasks use either `/tmp/test.c` or `/tmp/test.cpp` file as input.

*Note: To enable the `gcc` problem matcher, install the C/C++ Microsoft plugin and disable its intellisense to avoid conflicts with clangd's plugin (e.g. add `"C_Cpp.intelliSenseEngine": "disabled"` in `.vscode/settings`)*

1. **Step-By-Step Lowering Tasks:** \
 These tasks progressively lower the source code through various transformations, dumping each intermediate result in an output file in the `/tmp` directory: `test.raw`, `test.cir`, `test.mlir`, `test.ll`, `test.o`, and `test`. Individually, they are useful for testing a particular lowering stage (e.g. codegen, CIR to MLIR, MLIR to LLVM IR, etc.)
 - <u>C Preprocessing \| C++ Preprocessing</u>: Run the preprocessing stage on C and C++ source files, respectively.
 - <u>C to CIR (cc1) \| C++ to CIR (cc1)</u>: Convert preprocessed C/C++ code to Clang Intermediate Representation (CIR).
 - <u>CIR to MLIR (cir-opt)</u>: Transform CIR to the MLIR format.
 - <u>MLIR to LLVM (cir-translate)</u>: Translate MLIR code to LLVM IR.
 - <u>LLVM to Obj (llc)</u>: Compile LLVM IR to an object file.
 - <u>Obj to Exe (clang)</u>: Link the object file to produce an executable.
 - <u>Run</u>: Execute the produced executable and display the exit value.

1. **Multi-Step Tasks:**\
 Shortcuts for running a sequence of the lowering tasks mentioned above. Useful for generating all the intermediate results (CIR, MLIR, LLVM IR, etc) for a given C/C++ file.
 - <u>Run C:</u> Execute all steps starting at the `/tmp/test.c` file.
 - <u>Run C++:</u> Execute all steps starting at the `/tmp/test.cpp` file.
 - <u>Run LLVM Dialect:</u> Execute all steps starting at the `/tmp/test.mlir` file.
 - <u>Run LLVM IR:</u> Execute all steps starting at the `/tmp/test.ll` file.

1. **Miscellaneous Tasks**\
Some other useful tasks available in the JSON file:
 - <u>Debug Build</u>: Build only the `clang`, `cir-opt`, and `cir-translate` tools in debug mode. This is normally all you need to rebuild when developing for ClangIR, so use it to quickly test changes made to the code.
 - <u>Run Directly</u>: Execute a C/C++ file directly through ClangIR without dumping the intermediate results.
 - <u>Clang AST</u>: Dump the Abstract Syntax Tree (AST) of the source file for codegen development.
 - <u>Raw LLVM IR</u>: Produce an unoptimized reference LLVM IR code for comparison with ClangIR's LLVM IR result.
 - <u>LIT Tests</u>: Run all CIR LIT tests in the workspace to check if nothing broke.

#### Debugging

*Note: These debugging tasks depend on the "C Preprocessing" and "C++ Preprocessing" tasks from the previous section.*

Install LLDB (`sudo apt install lldb`) and the CodeLLDB VS Code plugin, then copy the [launch.json](../Files/launch.json) file into your `.vscode` folder. Attatching to child processes with LLDB is a bit difficult, so the debug configurations are made to be used for each in the lowering process:

1. <u>C++ to CIR</u>: Used to debug ClangIR C++ codegen.
1. <u>C to CIR</u>: Used to debug ClangIR C codegen.
1. <u>CIR to MLIR (cir-opt)</u>: Used to debug the `cir-opt` tool.
1. <u>MLIR to LLVM IR</u>: Used to debug the `cir-translate` tool.

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
