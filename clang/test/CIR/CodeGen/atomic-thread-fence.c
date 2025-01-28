// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// UN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// UN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s


struct Data {
  int value;
  void *ptr;
};

typedef struct Data *DataPtr;

void applyThreadFence() {
  __atomic_thread_fence(5);
}

// CIR-LABEL: cir.func no_proto @applyThreadFence
// CIR:   %0 = cir.const #cir.int<5> : !s32i
// CIR:   cir.atomic.fence(sync_scope = system, ordering = seq_cst)
// CIR:   cir.return

void applySignalFence() {
  __atomic_signal_fence(5);
}
// CIR-LABEL: cir.func no_proto @applySignalFence
// CIR:    %0 = cir.const #cir.int<5> : !s32i
// CIR:    cir.atomic.fence(sync_scope = single_thread, ordering = seq_cst)
// CIR:    cir.return

void modifyWithThreadFence(DataPtr d) {
  __atomic_thread_fence(5);
  d->value = 42;
}
// CIR-LABEL: cir.func @modifyWithThreadFence
// CIR:    %0 = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    cir.store %arg0, %0 : !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>
// CIR:    %1 = cir.const #cir.int<5> : !s32i
// CIR:    cir.atomic.fence(sync_scope = system, ordering = seq_cst)
// CIR:    %2 = cir.const #cir.int<42> : !s32i
// CIR:    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %4 = cir.get_member %3[0] {name = "value"} : !cir.ptr<!ty_Data> -> !cir.ptr<!s32i>
// CIR:    cir.store %2, %4 : !s32i, !cir.ptr<!s32i>
// CIR:    cir.return

void modifyWithSignalFence(DataPtr d) {
  __atomic_signal_fence(5);
  d->value = 24;
}
// CIR-LABEL: cir.func @modifyWithSignalFence
// CIR:    %0 = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    cir.store %arg0, %0 : !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>
// CIR:    %1 = cir.const #cir.int<5> : !s32i
// CIR:    cir.atomic.fence(sync_scope = single_thread, ordering = seq_cst)
// CIR:    %2 = cir.const #cir.int<24> : !s32i
// CIR:    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %4 = cir.get_member %3[0] {name = "value"} : !cir.ptr<!ty_Data> -> !cir.ptr<!s32i>
// CIR:    cir.store %2, %4 : !s32i, !cir.ptr<!s32i>
// CIR:    cir.return

void loadWithThreadFence(DataPtr d) {
  __atomic_thread_fence(5);
  __atomic_load_n(&d->ptr, 5);
}
// CIR-LABEL: cir.func @loadWithThreadFence
// CIR:    %0 = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    %1 = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["atomic-temp"] {alignment = 8 : i64}
// CIR:    cir.store %arg0, %0 : !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>
// CIR:    %2 = cir.const #cir.int<5> : !s32i
// CIR:    cir.atomic.fence(sync_scope = system, ordering = seq_cst)
// CIR:    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %4 = cir.get_member %3[1] {name = "ptr"} : !cir.ptr<!ty_Data> -> !cir.ptr<!cir.ptr<!void>>
// CIR:    %5 = cir.const #cir.int<5> : !s32i
// CIR:    %6 = cir.cast(bitcast, %4 : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    %7 = cir.load atomic(seq_cst) %6 : !cir.ptr<!u64i>, !u64i
// CIR:    %8 = cir.cast(bitcast, %1 : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    cir.store %7, %8 : !u64i, !cir.ptr<!u64i>
// CIR:    %9 = cir.load %1 : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.return

void loadWithSignalFence(DataPtr d) {
  __atomic_signal_fence(5);
  __atomic_load_n(&d->ptr, 5);
}
// CIR-LABEL: cir.func @loadWithSignalFence
// CIR:    %0 = cir.alloca !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>, ["d", init] {alignment = 8 : i64}
// CIR:    %1 = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["atomic-temp"] {alignment = 8 : i64}
// CIR:    cir.store %arg0, %0 : !cir.ptr<!ty_Data>, !cir.ptr<!cir.ptr<!ty_Data>>
// CIR:    %2 = cir.const #cir.int<5> : !s32i
// CIR:    cir.atomic.fence(sync_scope = single_thread, ordering = seq_cst)
// CIR:    %3 = cir.load %0 : !cir.ptr<!cir.ptr<!ty_Data>>, !cir.ptr<!ty_Data>
// CIR:    %4 = cir.get_member %3[1] {name = "ptr"} : !cir.ptr<!ty_Data> -> !cir.ptr<!cir.ptr<!void>>
// CIR:    %5 = cir.const #cir.int<5> : !s32i
// CIR:    %6 = cir.cast(bitcast, %4 : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    %7 = cir.load atomic(seq_cst) %6 : !cir.ptr<!u64i>, !u64i
// CIR:    %8 = cir.cast(bitcast, %1 : !cir.ptr<!cir.ptr<!void>>), !cir.ptr<!u64i>
// CIR:    cir.store %7, %8 : !u64i, !cir.ptr<!u64i>
// CIR:    %9 = cir.load %1 : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR:    cir.return
