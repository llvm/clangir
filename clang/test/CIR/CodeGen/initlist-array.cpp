// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir -clangir-disable-passes
// RUN: FileCheck --check-prefix=BEFORE --input-file=%t.cir %s

namespace std {
template <class b> class initializer_list {
  const b *array_start;
  const b *array_end;
};

} // namespace std

void f(std::initializer_list<int[2]>);
void test() {
  f({{1,2},{1,2}});
}

// BEFORE: [[INITLIST_TYPE:!.*]] = !cir.struct<class "std::initializer_list<int[2]>" {!cir.ptr<!cir.array<!s32i x 2>>, !cir.ptr<!cir.array<!s32i x 2>>} #cir.record.decl.ast>
// BEFORE: %0 = cir.alloca !ty_std3A3Ainitializer_list3Cint5B25D3E, !cir.ptr<!ty_std3A3Ainitializer_list3Cint5B25D3E>, ["agg.tmp0"] {alignment = 8 : i64}
// BEFORE: %1 = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["agg.tmp1"] {alignment = 4 : i64}
// BEFORE: %2 = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["agg.tmp2"] {alignment = 4 : i64}
// ignore cir for array initialization
// BEFORE: cir.std.initializer_list %0 (%1, %2 : !cir.ptr<!cir.array<!s32i x 2>>, !cir.ptr<!cir.array<!s32i x 2>>) : !cir.ptr<[[INITLIST_TYPE]]>
