// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir-enable -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir-enable -emit-cir -mmlir --mlir-print-ir-after=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=AFTER

#include "Inputs/std-compare-noncanonical.h"

auto three_way_strong(int x, int y) {
  return x <=> y;
}

// BEFORE: #cmp3way_info_strong_lt1eq2gt3_ = #cir.cmp3way_info<strong, lt = 1, eq = 2, gt = 3>
// BEFORE: cir.func @_Z16three_way_strongii
// BEFORE:   %{{.+}} = cir.cmp3way(%{{.+}} : !s32i, %{{.+}}, #cmp3way_info_strong_lt1eq2gt3_) : !s8i
// BEFORE: }

//      AFTER: #cmp3way_info_strong_ltn1eq0gt1_ = #cir.cmp3way_info<strong, lt = -1, eq = 0, gt = 1>
//      AFTER: cir.func @_Z16three_way_strongii
//      AFTER:   %[[#CMP3WAY_RESULT:]] = cir.cmp3way(%{{.+}} : !s32i, %{{.+}}, #cmp3way_info_strong_ltn1eq0gt1_) : !s8i
// AFTER-NEXT:   %[[#NEGONE:]] = cir.const(#cir.int<-1> : !s8i) : !s8i
// AFTER-NEXT:   %[[#ONE:]] = cir.const(#cir.int<1> : !s8i) : !s8i
// AFTER-NEXT:   %[[#CMP_TO_NEGONE:]] = cir.cmp(eq, %[[#CMP3WAY_RESULT]], %[[#NEGONE]]) : !s8i, !cir.bool
// AFTER-NEXT:   %[[#A:]] = cir.ternary(%[[#CMP_TO_NEGONE]], true {
// AFTER-NEXT:     cir.yield %[[#ONE]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#CMP3WAY_RESULT]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
// AFTER-NEXT:   %[[#ZERO:]] = cir.const(#cir.int<0> : !s8i) : !s8i
// AFTER-NEXT:   %[[#TWO:]] = cir.const(#cir.int<2> : !s8i) : !s8i
// AFTER-NEXT:   %[[#CMP_TO_ZERO:]] = cir.cmp(eq, %[[#A]], %[[#ZERO]]) : !s8i, !cir.bool
// AFTER-NEXT:   %[[#B:]] = cir.ternary(%[[#CMP_TO_ZERO]], true {
// AFTER-NEXT:     cir.yield %[[#TWO]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#A]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
// AFTER-NEXT:   %[[#ONE2:]] = cir.const(#cir.int<1> : !s8i) : !s8i
// AFTER-NEXT:   %[[#THREE:]] = cir.const(#cir.int<3> : !s8i) : !s8i
// AFTER-NEXT:   %[[#CMP_TO_ONE:]] = cir.cmp(eq, %[[#B]], %[[#ONE2]]) : !s8i, !cir.bool
// AFTER-NEXT:   %{{.+}} = cir.ternary(%[[#CMP_TO_ONE]], true {
// AFTER-NEXT:     cir.yield %[[#THREE]] : !s8i
// AFTER-NEXT:   }, false {
// AFTER-NEXT:     cir.yield %[[#B]] : !s8i
// AFTER-NEXT:   }) : (!cir.bool) -> !s8i
//      AFTER: }
