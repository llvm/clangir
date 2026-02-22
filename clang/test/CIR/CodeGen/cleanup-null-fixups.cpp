#include "std-cxx.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// CIR-LABEL: cir.func {{.*}} @_ZNK1d1eEv
// CIR-NEXT:   %[[V0:.*]] = cir.alloca !cir.ptr<!rec_d>, !cir.ptr<!cir.ptr<!rec_d>>, ["this", init]
// CIR-NEXT:   %[[V1:.*]] = cir.alloca !rec_a3A3Ac, !cir.ptr<!rec_a3A3Ac>, ["__retval"]
// CIR-NEXT:   %[[V2:.*]] = cir.alloca !rec_aj, !cir.ptr<!rec_aj>, ["ao"]
// CIR-NEXT:   cir.store %arg0, %[[V0]] : !cir.ptr<!rec_d>, !cir.ptr<!cir.ptr<!rec_d>>
// CIR-NEXT:   %[[V3:.*]] = cir.load %[[V0]] : !cir.ptr<!cir.ptr<!rec_d>>, !cir.ptr<!rec_d>
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     %[[V5:.*]] = cir.alloca !rec_aj, !cir.ptr<!rec_aj>, ["agg.tmp0"]
// CIR-NEXT:     %[[V6:.*]] = cir.get_global @an : !cir.ptr<!rec_aj>
// CIR-NEXT:     cir.copy %[[V6]] to %[[V5]] : !cir.ptr<!rec_aj>
// CIR-NEXT:     %[[V7:.*]] = cir.load align(1) %[[V5]] : !cir.ptr<!rec_aj>, !rec_aj
// CIR-NEXT:     cir.call @_ZN1a1cC1I2ajEET_(%[[V1]], %[[V7]]) : (!cir.ptr<!rec_a3A3Ac>, !rec_aj) -> ()
// CIR-NEXT:     cir.call @_ZN2ajD1Ev(%[[V5]]) : (!cir.ptr<!rec_aj>) -> ()
// CIR-NEXT:   }
// CIR-NEXT:   cir.call @_ZN2ajD1Ev(%[[V2]]) : (!cir.ptr<!rec_aj>) -> ()
// CIR-NEXT:   %[[V4:.*]] = cir.load align(1) %[[V1]] : !cir.ptr<!rec_a3A3Ac>, !rec_a3A3Ac
// CIR-NEXT:   cir.return %[[V4]] : !rec_a3A3Ac
// CIR-NEXT: }

// LLVM-LABEL: {{.*}} @_ZNK1d1eEv(ptr {{.*}})
// LLVM-NEXT:   %[[V2:.*]] = alloca %class.aj, i64 1, align 1
// LLVM-NEXT:   %[[V3:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   %[[V4:.*]] = alloca %"class.a::c", i64 1, align 1
// LLVM-NEXT:   %[[V5:.*]] = alloca %class.aj, i64 1, align 1
// LLVM-NEXT:   store ptr {{.*}}, ptr %[[V3]], align 8
// LLVM-NEXT:   %[[V6:.*]] = load ptr, ptr %[[V3]], align 8
// LLVM-NEXT:   br label %[[B7:.*]]
// LLVM: [[B7]]:
// LLVM-NEXT:   call void @llvm.memcpy.p0.p0.i32(ptr %[[V2]], ptr @an, i32 1, i1 false)
// LLVM-NEXT:   %[[V8:.*]] = load %class.aj, ptr %[[V2]], align 1
// LLVM-NEXT:   call void @_ZN1a1cC1I2ajEET_(ptr %[[V4]], %class.aj %[[V8]])
// LLVM-NEXT:   call void @_ZN2ajD1Ev(ptr %[[V2]])
// LLVM-NEXT:   br label %[[B9:.*]]
// LLVM: [[B9]]:
// LLVM-NEXT:   call void @_ZN2ajD1Ev(ptr %[[V5]])
// LLVM-NEXT:   %[[V10:.*]] = load %"class.a::c", ptr %[[V4]], align 1
// LLVM-NEXT:   ret %"class.a::c" %[[V10]]
// LLVM-NEXT: }

// OGCG-LABEL: {{.*}} @_ZNK1d1eEv(ptr {{.*}})
// OGCG:  %{{.*}} = alloca ptr, align 8
// OGCG-NEXT:  %{{.*}} = alloca ptr, align 8
// OGCG-NEXT:  %{{.*}} = alloca %{{.*}}, align 1
// OGCG-NEXT:  %{{.*}} = alloca %{{.*}}, align 1
// OGCG-NEXT:  store ptr %{{.*}}, ptr %{{.*}}, align 8
// OGCG-NEXT:  store ptr %{{.*}}, ptr %{{.*}}, align 8
// OGCG-NEXT:  %{{.*}} = load ptr, ptr %{{.*}}, align 8
// OGCG-NEXT:  call void @_ZN1a1cC1I2ajEET_(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}}, ptr noundef %{{.*}})
// OGCG-NEXT:  call void @_ZN2ajD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}})
// OGCG-NEXT:  call void @_ZN2ajD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}})
// OGCG-NEXT:  ret void
// OGCG-NEXT: }

inline namespace a {
class c {
public:
  template <typename b> c(b);
  ~c();
};
} // namespace a
class d {
  c e() const;
};
class aj {
public:
  ~aj();
} an;
c d::e() const {
  aj ao;
  return an;
  c(0);
}

// CIR-LABEL: cir.func {{.*}} @_Z3foov
// CIR-NEXT:  %[[V0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:  %[[V1:.*]] = cir.alloca !rec_std3A3Abasic_string3Cchar3E, !cir.ptr<!rec_std3A3Abasic_string3Cchar3E>, ["a", init]
// CIR-NEXT:  %[[V2:.*]] = cir.alloca !rec_std3A3Abasic_string3Cchar3E, !cir.ptr<!rec_std3A3Abasic_string3Cchar3E>, ["b"]
// CIR-NEXT:  cir.call @_ZNSbIcEC1Ev(%[[V1]]) : (!cir.ptr<!rec_std3A3Abasic_string3Cchar3E>) -> ()
// CIR-NEXT:  %[[V3:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:  cir.store align(4) %[[V3]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:  cir.call @_ZNSbIcED1Ev(%[[V1]]) : (!cir.ptr<!rec_std3A3Abasic_string3Cchar3E>) -> () extra(#fn_attr)
// CIR-NEXT:  %[[V4:.*]] = cir.load align(4) %[[V0]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:  cir.return %[[V4]] : !s32i
// CIR-NEXT: }

// LLVM-LABEL: {{.*}} @_Z3foov()
// LLVM-NEXT:   %[[V1:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   %[[V2:.*]] = alloca %"class.std::basic_string<char>", i64 1, align 1
// LLVM-NEXT:   %[[V3:.*]] = alloca %"class.std::basic_string<char>", i64 1, align 1
// LLVM-NEXT:   call void @_ZNSbIcEC1Ev(ptr %[[V2]])
// LLVM-NEXT:   store i32 0, ptr %[[V1]], align 4
// LLVM-NEXT:   call void @_ZNSbIcED1Ev(ptr %[[V2]])
// LLVM-NEXT:   %[[V4:.*]] = load i32, ptr %[[V1]], align 4
// LLVM-NEXT:   ret i32 %[[V4]]
// LLVM-NEXT: }

// OGCG-LABEL: {{.*}} @_Z3foov()
// OGCG:        %[[a:.*]] = alloca %"class.std::basic_string", align 1
// OGCG-NEXT:   %[[b:.*]] = alloca %"class.std::basic_string", align 1
// OGCG-NEXT:   call void @_ZNSbIcEC1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[a]])
// OGCG-NEXT:   call void @_ZNSbIcED1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[a]])
// OGCG-NEXT:   ret i32 0
// OGCG-NEXT: }

int foo() {
  std::string a;
  return 0;
  std::string b;
}
