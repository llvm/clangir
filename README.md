# ClangIR (CIR)

ClangIR is a high-level representation in Clang that reflects aspects of the C/C++
languages and their extensions. It is implemented using MLIR and occupies a position
between Clang's AST and LLVM IR.

The project is incubated under LLVM's GitHub
[umbrella](https://github.com/llvm) and is being [upstreamed](https://github.com/llvm/llvm-project/labels/ClangIR)
to the llvm-project (LLVM's official main repository).

GitHub incubator repo: [llvm/clangir](https://github.com/llvm/clangir).

Check out the [current status](https://llvm.github.io/clangir/Development/status.html)
for updates and [ClangIR in practice](https://llvm.github.io/clangir/Development/pipeline.html) for a tool overview. Also see
[motivation and history](https://llvm.github.io/clangir/Development/motivation.html)
for extra context about the project.

# Talks and posts

- Jun 2025: Asia LLVM. *ClangIR’s Footprint: A quick compile-time impact report*. [video](https://www.youtube.com/watch?v=Dh_RObp5SUE), [pdf](Files/asiallvm-brunolopes-25-cir-compile-time.pdf)
- Mar 2025: Cambridge Computer Science [Group Projects](https://group-projects.cst.cam.ac.uk/#__tabbed_1_2). Award on technical achievement. *[CUDA Support for ClangIR](https://wiki.cam.ac.uk/cl-design-projects/CUDA_Support_for_ClangIR)*. [video](https://www.youtube.com/watch?v=CO1w_L3yIbQ)
- Nov 2024: Supercomputing Workshop. *[Introducing ClangIR, High-Level IR for the C/C++ Family of Languages](https://sc24.supercomputing.org/proceedings/workshops/workshop_pages/misc312.html)*. [pdf](Files/sc24-clangir.pdf)
- Sep 2024: [GSoC 2024: Compile GPU kernels using ClangIR](https://blog.llvm.org/posts/2024-08-29-gsoc-opencl-c-support-for-clangir/)
- Sep 2024: [GSoC 2024: ABI Lowering in ClangIR](https://blog.llvm.org/posts/2024-09-07-abi-lowering-in-clangir/)
- Oct 2023: US LLVM Developers Meeting. *Evolution of ClangIR: A Year of Progress, Challenges, and Future Plans*. [video](https://www.youtube.com/watch?v=XNOPO3ogdfQ), [pdf](http://brunocardoso.cc/resources/2023-LLVMDevMtgClangIR.pdf).

### RFCs
- January 2024: [RFC: Upstreaming ClangIR](https://discourse.llvm.org/t/rfc-upstreaming-clangir/76587)
- June 2022: [RFC: An MLIR based Clang IR (CIR)](https://discourse.llvm.org/t/rfc-an-mlir-based-clang-ir-cir/63319)

# Where to go from here?

Check out the docs for [contributing to the
project](https://llvm.github.io/clangir/GettingStarted/contrib.html) and dive
into the [CIR Dialect](https://llvm.github.io/clangir/Dialect/),
list of [CIR passes](https://llvm.github.io/clangir/Dialect/passes.html) in Clang or the ClangIR [pipeline](https://llvm.github.io/clangir/Development/pipeline.html).

### Inspiration

ClangIR is inspired in the success of other languages that greatly benefit from
a middle-level IR, such as
[Swift](https://apple-swift.readthedocs.io/en/latest/SIL.html) and
[Rust](https://rustc-dev-guide.rust-lang.org/mir/index.html). Particularly,
optionally attaching AST nodes to CIR operations is inspired by SIL references
to AST nodes in Swift.

<!---
On vim use ":r!date"
-->

*Last updated: Mon Aug 25 21:50:24 PDT 2025*
