// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o %t.cir
// RUN: FileCheck --input-file=%t.cir -check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --input-file=%t.ll -check-prefix=LLVM %s

// TODO: This file is inspired by clang/test/CodeGenCXX/new.cpp, add all tests from it.

typedef __typeof__(sizeof(0)) size_t;

// Declare an 'operator new' template to tickle a bug in __builtin_operator_new.
template<typename T> void *operator new(size_t, int (*)(T));

// Ensure that this declaration doesn't cause operator new to lose its
// 'noalias' attribute.
void *operator new[](size_t);

namespace std {
  struct nothrow_t {};
  enum class align_val_t : size_t { __zero = 0,
                                  __max = (size_t)-1 };
}
std::nothrow_t nothrow;

// Declare the reserved placement operators.
void *operator new(size_t, void*) throw();
void operator delete(void*, void*) throw();
void *operator new[](size_t, void*) throw();
void operator delete[](void*, void*) throw();

// Declare the replaceable global allocation operators.
void *operator new(size_t, const std::nothrow_t &) throw();
void *operator new[](size_t, const std::nothrow_t &) throw();
void operator delete(void *, const std::nothrow_t &) throw();
void operator delete[](void *, const std::nothrow_t &) throw();

// Declare some other placemenet operators.
void *operator new(size_t, void*, bool) throw();
void *operator new[](size_t, void*, bool) throw();

namespace test15 {
  struct A { A(); ~A(); };
  // CIR-DAG:   ![[TEST15A:.*]] = !cir.struct<struct "test15::A" {!u8i}

  void test0a(void *p) {
    new (p) A();
  }

  // CIR-LABEL:   cir.func @_ZN6test156test0bEPv(
  // CIR-SAME:                                   %[[VAL_0:.*]]: !cir.ptr<!void>
  // CIR:           %[[VAL_1:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["p", init] {alignment = 8 : i64}
  // CIR:           cir.store %[[VAL_0]], %[[VAL_1]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
  // CIR:           %[[VAL_2:.*]] = cir.const #cir.int<1> : !u64i
  // CIR:           %[[VAL_3:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR:           %[[VAL_4:.*]] = cir.const #true
  // CIR:           %[[VAL_5:.*]] = cir.call @_ZnwmPvb(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
  // CIR:           %[[VAL_6:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
  // CIR:           %[[VAL_7:.*]] = cir.cmp(ne, %[[VAL_5]], %[[VAL_6]]) : !cir.ptr<!void>, !cir.bool
  // CIR:           %[[VAL_8:.*]] = cir.cast(bitcast, %[[VAL_5]] : !cir.ptr<!void>), !cir.ptr<![[TEST15A]]>
  // CIR:           cir.if %[[VAL_7]] {
  // CIR:             cir.call @_ZN6test151AC1Ev(%[[VAL_8]]) : (!cir.ptr<![[TEST15A]]>) -> ()
  // CIR:           }
  // CIR:           cir.return
  // CIR:         }

  // LLVM-LABEL: _ZN6test156test0bEPv
  // LLVM:         %[[VAL_0:.*]] = alloca ptr, i64 1, align 8
  // LLVM:         store ptr %[[VAL_1:.*]], ptr %[[VAL_0]], align 8
  // LLVM:         %[[VAL_2:.*]] = load ptr, ptr %[[VAL_0]], align 8
  // LLVM:         %[[VAL_3:.*]] = call ptr @_ZnwmPvb(i64 1, ptr %[[VAL_2]], i1 true)
  // LLVM:         %[[VAL_4:.*]] = icmp ne ptr %[[VAL_3]], null
  // LLVM:         br i1 %[[VAL_4]], label %[[VAL_5:.*]], label %[[VAL_6:.*]]
  // LLVM:       [[VAL_5]]:                                                ; preds = %[[VAL_7:.*]]
  // LLVM:         call void @_ZN6test151AC1Ev(ptr %[[VAL_3]])
  // LLVM:         br label %[[VAL_6]]
  // LLVM:       [[VAL_6]]:                                                ; preds = %[[VAL_5]], %[[VAL_7]]
  // LLVM:         ret void

  void test0b(void *p) {
    new (p, true) A();
  }
}

extern "C" void test_basic() {
  __builtin_operator_delete(__builtin_operator_new(4));
  // CIR-LABEL: cir.func @test_basic
  // CIR: [[P:%.*]] = cir.call @_Znwm({{%.*}}) : (!u64i) -> !cir.ptr<!void>
  // CIR: cir.call @_ZdlPv([[P]]) : (!cir.ptr<!void>) -> ()
  // CIR: cir.return

  // LLVM-LABEL: define{{.*}} void @test_basic()
  // LLVM: [[P:%.*]] = call ptr @_Znwm(i64 4)
  // LLVM: call void @_ZdlPv(ptr [[P]])
  // LLVM: ret void
}

extern "C" void test_aligned_alloc() {
  __builtin_operator_delete(__builtin_operator_new(4, std::align_val_t(4)), std::align_val_t(4));

  // CIR-LABEL: cir.func @test_aligned_alloc
  // CIR: [[P:%.*]] = cir.call @_ZnwmSt11align_val_t({{%.*}}, {{%.*}}) : (!u64i, !u64i) -> !cir.ptr<!void>
  // CIR: cir.call @_ZdlPvSt11align_val_t([[P]], {{%.*}}) : (!cir.ptr<!void>, !u64i) -> ()
  // CIR: cir.return

  // LLVM-LABEL: define{{.*}} void @test_aligned_alloc()
  // LLVM: [[P:%.*]] = call ptr @_ZnwmSt11align_val_t(i64 4, i64 4)
  // LLVM: call void @_ZdlPvSt11align_val_t(ptr [[P]], i64 4)
  // LLVM: ret void
}

extern "C" void test_sized_delete() {
  __builtin_operator_delete(__builtin_operator_new(4), 4);

  // CIR-LABEL: cir.func @test_sized_delete
  // CIR: [[P:%.*]] = cir.call @_Znwm({{%.*}}) : (!u64i) -> !cir.ptr<!void>
  // CIR: cir.call @_ZdlPvm([[P]], {{%.*}}) : (!cir.ptr<!void>, !u64i) -> ()
  // CIR: cir.return

  // LLVM-LABEL: define{{.*}} void @test_sized_delete()
  // LLVM: [[P:%.*]] = call ptr @_Znwm(i64 4)
  // LLVM: call void @_ZdlPvm(ptr [[P]], i64 4)
  // LLVM: ret void
}
