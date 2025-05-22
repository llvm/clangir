// Check that the `cir.triple` attribute is correctly translated into the LLVM `llvm.triple` attribute.
// This test cannot use `-triple x86_64-unknown-linux-gnu` explicitly,
// as the frontend must infer the triple based on the host.
// Restricting this test to supported targets until CIR supports more platforms.

// RUN: %clang -fclangir -fno-clangir-direct-lowering -c %s -o %t

// TODO: Relax target constraint once CIR supports additional backends.
// REQUIRES: target=x86_64{{.*}}-linux{{.*}}


int test_target_triple_passthrough() {
    // Variables
    int number = 0;

    return number;
} 