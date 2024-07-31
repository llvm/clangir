// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

namespace std {
template <class b> class initializer_list {
  const b *c;
  unsigned long len;
};
template <class b>
void f(initializer_list<b>) {;}
void test() {
  f({7});
}
} // namespace std

// CIR: [[INITLIST_TYPE:!.*]] = !cir.struct<class "std::initializer_list<int>" {!cir.ptr<!cir.int<s, 32>>, !cir.int<u, 64>}>

// CIR: cir.func linkonce_odr @_ZSt1fIiEvSt16initializer_listIT_E(%arg0: [[INITLIST_TYPE]]
// CIR: [[REG0:%.*]] = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>,
// CIR: cir.store %arg0, [[REG0]] : [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>
// CIR: cir.return

// CIR: cir.func @_ZSt4testv()
// CIR: cir.scope {
// CIR: [[LIST_PTR:%.*]] = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>,
// CIR: [[ARRAY:%.*]] = cir.alloca !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>,
// CIR: [[DECAY_PTR:%.*]] = cir.cast(array_to_ptrdecay, [[ARRAY]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
// CIR: [[SEVEN:%.*]] = cir.const #cir.int<7> : !s32i
// CIR: cir.store [[SEVEN]], [[DECAY_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: [[FLD_C:%.*]] = cir.get_member [[LIST_PTR]][0] {name = "c"} : !cir.ptr<[[INITLIST_TYPE]]> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR: [[ARRAY_PTR:%.*]] = cir.cast(bitcast, [[FLD_C]] : !cir.ptr<!cir.ptr<!s32i>>), !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>
// CIR: cir.store [[ARRAY]], [[ARRAY_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>
// CIR: [[LENGTH_ONE:%.*]] = cir.const #cir.int<1>
// CIR: [[FLD_LEN:%.*]] = cir.get_member [[LIST_PTR]][1] {name = "len"} : !cir.ptr<[[INITLIST_TYPE]]> -> !cir.ptr<!u64i>
// CIR: cir.store [[LENGTH_ONE]], [[FLD_LEN]] : !u64i, !cir.ptr<!u64i>
// CIR: [[ARG2PASS:%.*]] = cir.load [[LIST_PTR]] : !cir.ptr<[[INITLIST_TYPE]]>,  [[INITLIST_TYPE]]
// CIR: cir.call @_ZSt1fIiEvSt16initializer_listIT_E([[ARG2PASS]]) : ([[INITLIST_TYPE]]) -> ()
// CIR: }
// CIR: cir.return
// CIR: }

// LLVM: %"class.std::initializer_list<int>" = type { ptr, i64 }
