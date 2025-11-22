// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CHECK-EBO-BASIC

// Test for Empty Base Optimization (EBO) with inheriting constructors
// This was causing assertion failure in CIRGenRecordLayout::getCIRFieldNo()
// Bug: Fields optimized away by EBO don't exist in FieldInfo map
//
// The issue occurs when all three conditions are met:
// 1. A class has a member that's an empty type
// 2. The class uses inheriting constructors (using Base::Base)
// 3. The inherited constructor is instantiated
//
// Without the fix, this would crash with:
//   Assertion `FieldInfo.count(FD) && "Invalid field for record!"' failed
//   at CIRGenRecordLayout.h:187

struct Empty {
    constexpr Empty() noexcept = default;
};

struct HasEmptyMember {
    int* ptr;
    Empty e;
    HasEmptyMember() = default;
};

struct DerivedWithInheriting : HasEmptyMember {
    using HasEmptyMember::HasEmptyMember;  // Inheriting constructor
};

void test_ebo_inheriting_ctor() {
    DerivedWithInheriting d;  // This used to crash
}

// CHECK-EBO-BASIC: cir.func{{.*}}@_Z24test_ebo_inheriting_ctorv
// CHECK-EBO-BASIC:   %{{.*}} = cir.alloca !rec_DerivedWithInheriting
// CHECK-EBO-BASIC:   cir.return

int main() {
    test_ebo_inheriting_ctor();
    return 0;
}

// CHECK-EBO-BASIC: cir.func{{.*}}@main
// CHECK-EBO-BASIC:   cir.call @_Z24test_ebo_inheriting_ctorv()
// CHECK-EBO-BASIC:   cir.return
