---
sort : 4
---

## Benchmarks

The tables below show the number and percentage of passing tests, for each benchmark category and in different mode of configurations of ClangIR.
The pipeline is: AST -> CIR -> CIR Passes -> LLVM -> -O2 opt ...

### x86_64
- Target: Linux, x86_64
- Host: AMD EPYC-Milan Processor, 166 cores, 256GB RAM, CentOS 9.0
- Build mode: Release (no asserts)
- Compiler flags used: '-O2'

#### spec2017int

| Configuration | 2025-08 |
|---------------|----------|
| cir-incubator | 6 (60.00%) |
| cir-upstream | 1 (10.00%) |
| no-cir | 10 (100.00%) |

#### multisource

| Configuration | 2025-08 |
|---------------|----------|
| cir-incubator | 167 (83.08%) |
| cir-upstream | 49 (24.38%) |
| no-cir | 201 (100.00%) |

#### singlesource

| Configuration | 2025-08 |
|---------------|----------|
| cir-incubator | 1673 (91.27%) |
| cir-upstream | 1203 (65.63%) |
| no-cir | 1832 (99.95%) |

### ARM64
TBD
