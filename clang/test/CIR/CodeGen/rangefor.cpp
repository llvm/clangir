// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

typedef enum enumy {
  Unknown = 0,
  Some = 1000024002,
} enumy;

typedef struct triple {
  enumy type;
  void* __attribute__((__may_alias__)) next;
  unsigned image;
} triple;

void init(unsigned numImages) {
  std::vector<triple> images(numImages);
  for (auto& image : images) {
    image = {Some};
  }
}

// CHECK-DAG: !rec_triple = !cir.record<struct "triple" {!u32i, !cir.ptr<!void>, !u32i}>
// CHECK-DAG: ![[VEC:.*]] = !cir.record<class "std::vector<triple>" {!cir.ptr<!rec_triple>, !cir.ptr<!rec_triple>, !cir.ptr<!rec_triple>}>
// CHECK-DAG: ![[VEC_IT:.*]] = !cir.record<struct "__vector_iterator<triple, triple *, triple &>" {!cir.ptr<!rec_triple>}>

// CHECK: cir.func dso_local @_Z4initj(%arg0: !u32i
// CHECK:   %0 = cir.alloca !u32i, !cir.ptr<!u32i>, ["numImages", init] {alignment = 4 : i64}
// CHECK:   %1 = cir.alloca ![[VEC]], !cir.ptr<![[VEC]]>, ["images", init] {alignment = 8 : i64}
// CHECK:   cir.store{{.*}} %arg0, %0 : !u32i, !cir.ptr<!u32i>
// CHECK:   %2 = cir.load{{.*}} %0 : !cir.ptr<!u32i>, !u32i
// CHECK:   %3 = cir.cast(integral, %2 : !u32i), !u64i
// CHECK:   cir.call @_ZNSt6vectorI6tripleEC1Em(%1, %3) : (!cir.ptr<![[VEC]]>, !u64i) -> ()
// CHECK:   cir.scope {
// CHECK:     %4 = cir.alloca !cir.ptr<![[VEC]]>, !cir.ptr<!cir.ptr<![[VEC]]>>, ["__range1", init, const] {alignment = 8 : i64}
// CHECK:     %5 = cir.alloca ![[VEC_IT]], !cir.ptr<![[VEC_IT]]>, ["__begin1", init] {alignment = 8 : i64}
// CHECK:     %6 = cir.alloca ![[VEC_IT]], !cir.ptr<![[VEC_IT]]>, ["__end1", init] {alignment = 8 : i64}
// CHECK:     %7 = cir.alloca !cir.ptr<!rec_triple>, !cir.ptr<!cir.ptr<!rec_triple>>, ["image", init, const] {alignment = 8 : i64}
// CHECK:     cir.store{{.*}} %1, %4 : !cir.ptr<![[VEC]]>, !cir.ptr<!cir.ptr<![[VEC]]>>
// CHECK:     %8 = cir.load{{.*}} %4 : !cir.ptr<!cir.ptr<![[VEC]]>>, !cir.ptr<![[VEC]]>
// CHECK:     %9 = cir.call @_ZNSt6vectorI6tripleE5beginEv(%8) : (!cir.ptr<![[VEC]]>) -> ![[VEC_IT]]
// CHECK:     cir.store{{.*}} %9, %5 : ![[VEC_IT]], !cir.ptr<![[VEC_IT]]>
// CHECK:     %10 = cir.load{{.*}} %4 : !cir.ptr<!cir.ptr<![[VEC]]>>, !cir.ptr<![[VEC]]>
// CHECK:     %11 = cir.call @_ZNSt6vectorI6tripleE3endEv(%10) : (!cir.ptr<![[VEC]]>) -> ![[VEC_IT]]
// CHECK:     cir.store{{.*}} %11, %6 : ![[VEC_IT]], !cir.ptr<![[VEC_IT]]>
// CHECK:     cir.for : cond {
// CHECK:       %12 = cir.call @_ZNK17__vector_iteratorI6triplePS0_RS0_EneERKS3_(%5, %6) : (!cir.ptr<![[VEC_IT]]>, !cir.ptr<![[VEC_IT]]>) -> !cir.bool
// CHECK:       cir.condition(%12)
// CHECK:     } body {
// CHECK:       %12 = cir.call @_ZNK17__vector_iteratorI6triplePS0_RS0_EdeEv(%5) : (!cir.ptr<![[VEC_IT]]>) -> !cir.ptr<!rec_triple>
// CHECK:       cir.store{{.*}} %12, %7 : !cir.ptr<!rec_triple>, !cir.ptr<!cir.ptr<!rec_triple>>
// CHECK:       cir.scope {
// CHECK:         %13 = cir.alloca !rec_triple, !cir.ptr<!rec_triple>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:         %14 = cir.const #cir.zero : !rec_triple
// CHECK:         cir.store{{.*}} %14, %13 : !rec_triple, !cir.ptr<!rec_triple>
// CHECK:         %15 = cir.get_member %13[0] {name = "type"} : !cir.ptr<!rec_triple> -> !cir.ptr<!u32i>
// CHECK:         %16 = cir.const #cir.int<1000024002> : !u32i
// CHECK:         cir.store{{.*}} %16, %15 : !u32i, !cir.ptr<!u32i>
// CHECK:         %17 = cir.get_member %13[1] {name = "next"} : !cir.ptr<!rec_triple> -> !cir.ptr<!cir.ptr<!void>>
// CHECK:         %18 = cir.get_member %13[2] {name = "image"} : !cir.ptr<!rec_triple> -> !cir.ptr<!u32i>
// CHECK:         %19 = cir.load{{.*}} %7 : !cir.ptr<!cir.ptr<!rec_triple>>, !cir.ptr<!rec_triple>
// CHECK:         %20 = cir.call @_ZN6tripleaSEOS_(%19, %13) : (!cir.ptr<!rec_triple>, !cir.ptr<!rec_triple>) -> !cir.ptr<!rec_triple>
// CHECK:       }
// CHECK:       cir.yield
// CHECK:     } step {
// CHECK:       %12 = cir.call @_ZN17__vector_iteratorI6triplePS0_RS0_EppEv(%5) : (!cir.ptr<![[VEC_IT]]>) -> !cir.ptr<![[VEC_IT]]>
// CHECK:       cir.yield
// CHECK:     }
// CHECK:   }
// CHECK:   cir.return
