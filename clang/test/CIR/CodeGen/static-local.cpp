// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int fnA();

void foo() {
  static int val = fnA();
}

//      CHECK: @_ZZ3foovE3val = internal global i32 0, align 4
//      CHECK: @_ZGVZ3foovE3val = internal global i64 0, align 8
//      CHECK: declare i32 @_Z3fnAv()
//      CHECK: declare i32 @__cxa_guard_acquire(ptr)
//      CHECK: declare void @__cxa_guard_release(ptr)

//      CHECK: define dso_local void @_Z3foov()
// CHECK-NEXT:   %1 = load atomic i8, ptr @_ZGVZ3foovE3val acquire, align 1
// CHECK-NEXT:   %2 = icmp eq i8 %1, 0
// CHECK-NEXT:   br i1 %2, label %3, label %8
//  CHECK-DAG: 3:
// CHECK-NEXT:   %4 = call i32 @__cxa_guard_acquire(ptr @_ZGVZ3foovE3val)
// CHECK-NEXT:   %5 = icmp ne i32 %4, 0
// CHECK-NEXT:   br i1 %5, label %6, label %8
//  CHECK-DAG: 6:
// CHECK-NEXT:   %7 = call i32 @_Z3fnAv()
// CHECK-NEXT:   store i32 %7, ptr @_ZZ3foovE3val, align 4
// CHECK-NEXT:   call void @__cxa_guard_release(ptr @_ZGVZ3foovE3val)
// CHECK-NEXT:   br label %8
//  CHECK-DAG: 8:
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
