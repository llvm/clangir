// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir  %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm  %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm  %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void A(void) {
  void *ptr = &&A;
A:
  return;
}
// CIR:  cir.func dso_local @A
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    [[BLOCK:%.*]] = cir.blockaddress <@A, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.br ^bb1
// CIR:  ^bb1:  // pred: ^bb0
// CIR:    cir.label "A"
// CIR:    cir.return
//
// LLVM: define dso_local void @A()
// LLVM:   [[PTR:%.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr blockaddress(@A, %[[A:.*]]), ptr [[PTR]], align 8
// LLVM:   br label %[[A]]
// LLVM: [[A]]:                                                ; preds = %0
// LLVM:   ret void

// OGCG: define dso_local void @A()
// OGCG:   [[PTR:%.*]] = alloca ptr, align 8
// OGCG:   store ptr blockaddress(@A, %A), ptr [[PTR]], align 8
// OGCG:   br label %A
// OGCG: A:                                                ; preds = %entry, %indirectgoto
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; No predecessors!
// OGCG:   indirectbr ptr poison, [label %A]

void B(void) {
B:
  void *ptr = &&B;
}

// CIR:  cir.func dso_local @B()
// CIR:    [[PTR:%.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init] {alignment = 8 : i64}
// CIR:    cir.br ^bb1
// CIR:   ^bb1:  // pred: ^bb0
// CIR:    cir.label "B"
// CIR:    [[BLOCK:%.*]] = cir.blockaddress <@B, "B"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[BLOCK]], [[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return

// LLVM: define dso_local void @B
// LLVM:   %[[PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   br label %[[B:.*]]
// LLVM: [[B]]:
// LLVM:   store ptr blockaddress(@B, %[[B]]), ptr %[[PTR]], align 8
// LLVM:   ret void

// OGCG: define dso_local void @B
// OGCG:   [[PTR:%.*]] = alloca ptr, align 8
// OGCG:   br label %B
// OGCG: B:                                                ; preds = %indirectgoto, %entry
// OGCG:   store ptr blockaddress(@B, %B), ptr [[PTR]], align 8
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; No predecessors!
// OGCG:   indirectbr ptr poison, [label %B]

void C(int x) {
    void *ptr = (x == 0) ? &&A : &&B;
A:
    return;
B:
    return;
}

// CIR:  cir.func dso_local @C
// CIR:    [[BLOCK1:%.*]] = cir.blockaddress <@C, "A"> -> !cir.ptr<!void>
// CIR:    [[BLOCK2:%.*]] = cir.blockaddress <@C, "B"> -> !cir.ptr<!void>
// CIR:    [[COND:%.*]] = cir.select if [[CMP:%.*]] then [[BLOCK1]] else [[BLOCK2]] : (!cir.bool, !cir.ptr<!void>, !cir.ptr<!void>) -> !cir.ptr<!void>
// CIR:    cir.store align(8) [[COND]], [[PTR:%.*]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.br ^bb2
// CIR:  ^bb1:  // 2 preds: ^bb2, ^bb3
// CIR:    cir.return
// CIR:  ^bb2:  // pred: ^bb0
// CIR:    cir.label "A"
// CIR:    cir.br ^bb1
// CIR:  ^bb3:  // no predecessors
// CIR:    cir.label "B"
// CIR:    cir.br ^bb1

// LLVM: define dso_local void @C(i32 %0)
// LLVM:   [[COND:%.*]] = select i1 [[CMP:%.*]], ptr blockaddress(@C, %[[A:.*]]), ptr blockaddress(@C, %[[B:.*]])
// LLVM:   store ptr [[COND]], ptr [[PTR:%.*]], align 8
// LLVM:   br label %[[A]]
// LLVM: [[RET:.*]]:
// LLVM:   ret void
// LLVM: [[A]]:
// LLVM:   br label %[[RET]]
// LLVM: [[B]]:
// LLVM:   br label %[[RET]]

// OGCG: define dso_local void @C
// OGCG:   [[COND:%.*]] = select i1 [[CMP:%.*]], ptr blockaddress(@C, %A), ptr blockaddress(@C, %B)
// OGCG:   store ptr [[COND]], ptr [[PTR:%.*]], align 8
// OGCG:   br label %A
// OGCG: A:                                                ; preds = %entry, %indirectgoto
// OGCG:   br label %return
// OGCG: B:                                                ; preds = %indirectgoto
// OGCG:   br label %return
// OGCG: return:                                           ; preds = %B, %A
// OGCG:   ret void
// OGCG: indirectgoto:                                     ; No predecessors!
// OGCG:   indirectbr ptr poison, [label %A, label %B]

void D(void) {
  void *ptr = &&A;
  void *ptr2 = &&A;
A:
  void *ptr3 = &&A;
  return;
}

// CIR:  cir.func dso_local @D
// CIR:    %[[PTR:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr", init]
// CIR:    %[[PTR2:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr2", init]
// CIR:    %[[PTR3:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["ptr3", init]
// CIR:    %[[BLK1:.*]] = cir.blockaddress <@D, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK1]], %[[PTR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[BLK2:.*]] = cir.blockaddress <@D, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK2]], %[[PTR2]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.br ^bb1
// CIR:  ^bb1:  // pred: ^bb0
// CIR:    cir.label "A"
// CIR:    %[[BLK3:.*]] = cir.blockaddress <@D, "A"> -> !cir.ptr<!void>
// CIR:    cir.store align(8) %[[BLK3]], %[[PTR3]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR:    cir.return

// LLVM: define dso_local void @D
// LLVM:   %[[PTR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PTR2:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[PTR3:.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr blockaddress(@D, %[[A:.*]]), ptr %[[PTR]], align 8
// LLVM:   store ptr blockaddress(@D, %[[A]]), ptr %[[PTR2]], align 8
// LLVM:   br label %[[A]]
// LLVM: [[A]]:
// LLVM:   store ptr blockaddress(@D, %[[A]]), ptr %[[PTR3]], align 8
// LLVM:   ret void

// OGCG: define dso_local void @D
// OGCG:   %[[PTR:.*]] = alloca ptr, align 8
// OGCG:   %[[PTR2:.*]] = alloca ptr, align 8
// OGCG:   %[[PTR3:.*]] = alloca ptr, align 8
// OGCG:   store ptr blockaddress(@D, %A), ptr %[[PTR]], align 8
// OGCG:   store ptr blockaddress(@D, %A), ptr %[[PTR2]], align 8
// OGCG:   br label %A
// OGCG: A:
// OGCG:   store ptr blockaddress(@D, %A), ptr %[[PTR3]], align 8
// OGCG:   ret void
// OGCG: indirectgoto:
// OGCG:   indirectbr ptr poison, [label %A, label %A, label %A]
