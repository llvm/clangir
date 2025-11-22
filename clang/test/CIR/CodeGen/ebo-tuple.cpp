// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG

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

// CIR: cir.func{{.*}}@_Z24test_ebo_inheriting_ctorv
// CIR:   %{{.*}} = cir.alloca !rec_DerivedWithInheriting
// CIR:   cir.return

// LLVM: define {{.*}}@_Z24test_ebo_inheriting_ctorv
// LLVM:   alloca %struct.DerivedWithInheriting
// LLVM:   ret void

// OGCG: define {{.*}}@_Z24test_ebo_inheriting_ctorv
// OGCG:   alloca %struct.DerivedWithInheriting
// OGCG:   ret void

int main() {
    test_ebo_inheriting_ctor();
    return 0;
}

// CIR: cir.func{{.*}}@main
// CIR:   cir.call @_Z24test_ebo_inheriting_ctorv()
// CIR:   cir.return

// LLVM: define {{.*}}@main
// LLVM:   call {{.*}}@_Z24test_ebo_inheriting_ctorv()
// LLVM:   ret i32

// OGCG: define {{.*}}@main
// OGCG:   call {{.*}}@_Z24test_ebo_inheriting_ctorv()
// OGCG:   ret i32
