// Clang returns 1 when wrong arguments are given.
// RUN: not %clang_cc1 -mmlir -mlir-disable-threadingd  -mmlir -mlir-print-op-genericd 2>&1 | FileCheck %s --check-prefix=WRONG
// Test that the driver can pass mlir args to cc1.
// RUN: %clang -### -mmlir -mlir-disable-threading %s 2>&1 | FileCheck %s --check-prefix=CC1


// WRONG: ClangIR (MLIR option parsing): Unknown command line argument '-mlir-disable-threadingd'.  Try: 'ClangIR (MLIR option parsing) --help'
// WRONG: ClangIR (MLIR option parsing): Did you mean '--mlir-disable-threading'?
// WRONG: ClangIR (MLIR option parsing): Unknown command line argument '-mlir-print-op-genericd'.  Try: 'ClangIR (MLIR option parsing) --help'
// WRONG: ClangIR (MLIR option parsing): Did you mean '--mlir-print-op-generic'?

// CC1: "-mmlir" "-mlir-disable-threading"
