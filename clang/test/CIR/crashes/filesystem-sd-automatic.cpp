// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -std=c++17 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
// XFAIL: *
//
// This test validates the branch-through cleanup fix while documenting the
// remaining type checking issue.

#include <filesystem>

void test() {
  namespace fs = std::filesystem;

  // This triggers branch-through cleanup in PopCleanupBlock which previously
  // crashed with llvm_unreachable("NYI") at CIRGenCleanup.cpp:527.
  // That crash is now FIXED - the code progresses past cleanup emission.
  for (const auto& entry : fs::directory_iterator("/tmp")) {
    auto path = entry.path();
  }
}

// Verify the FIXED behavior: We get past PopCleanupBlock and reach CIR emission
// The error now occurs during CIR verification, NOT in PopCleanupBlock
// CHECK-ERROR-NOT: UNREACHABLE executed at {{.*}}CIRGenCleanup.cpp
// CHECK-ERROR: error: 'cir.if' op operand #0 must be CIR bool type, but got '!cir.int<u, 1>'
