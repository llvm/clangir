---
sort : 5
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

| Configuration | 2025-08-21 | 2025-09-03 |
|---------------|----------|----------|
| cir-incubator | 6 (60.00%) | 6 (60.00%) |
| cir-incubator-callconv | 6 (60.00%) | 6 (60.00%) |
| cir-incubator-throughmlir | 0 (0) | 0 (0) |
| cir-upstream | 1 (10.00%) | 1 (10.00%) |
| no-cir | 10 (100.00%) | 10 (100.00%) |

#### spec2017fp

| Configuration | 2025-08-21 | 2025-09-03 |
|---------------|----------|----------|
| cir-incubator | 4 (50.00%) | 4 (50.00%) |
| cir-incubator-callconv | 3 (37.50%) | 3 (37.50%) |
| cir-incubator-throughmlir | 0 (0) | 0 (0) |
| cir-upstream | 1 (12.50%) | 1 (12.50%) |
| no-cir | 8 (100.00%) | 8 (100.00%) |

#### multisource

| Configuration | 2025-08-21 | 2025-09-03 |
|---------------|----------|----------|
| cir-incubator | 167 (83.08%) | 167 (83.08%) |
| cir-incubator-callconv | 148 (73.63%) | 148 (73.63%) |
| cir-incubator-throughmlir | 0 (0) | 0 (0) |
| cir-upstream | 49 (24.38%) | 59 (29.35%) |
| no-cir | 201 (100.00%) | 201 (100.00%) |

#### singlesource

| Configuration | 2025-08-21 | 2025-09-03 |
|---------------|----------|----------|
| cir-incubator | 1673 (91.27%) | 1674 (91.33%) |
| cir-incubator-callconv | 1642 (89.58%) | 1643 (89.63%) |
| cir-incubator-throughmlir | 472 (25.75%) | 472 (25.75%) |
| cir-upstream | 1203 (65.63%) | 1248 (68.09%) |
| no-cir | 1832 (99.95%) | 1832 (99.95%) |

### ARM64
TBD
