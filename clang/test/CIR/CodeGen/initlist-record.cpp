// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir -clangir-disable-passes
// RUN: FileCheck --check-prefix=BEFORE --input-file=%t.cir %s

namespace std {
template <class b> class initializer_list {
  const b *array_start;
  const b *array_end;
};

} // namespace std

struct A {};
void f(std::initializer_list<A>);
void test() {
  f({A{},{}});
}

// BEFORE: [[INITLIST_TYPE:!.*]] = !cir.struct<class "std::initializer_list<A>" {!cir.ptr<!ty_A>, !cir.ptr<!ty_A>} #cir.record.decl.ast>
// BEFORE: %0 = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>, ["agg.tmp0"] {alignment = 8 : i64}
// BEFORE: %1 = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["agg.tmp1"] {alignment = 1 : i64}
// BEFORE: %2 = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["agg.tmp2"] {alignment = 1 : i64}
// BEFORE: cir.std.initializer_list %0 (%1, %2 : !cir.ptr<!ty_A>, !cir.ptr<!ty_A>) : !cir.ptr<[[INITLIST_TYPE]]>
// BEFORE: %3 = cir.load %0 : !cir.ptr<[[INITLIST_TYPE]]>, [[INITLIST_TYPE]]
// BEFORE: cir.call @_Z1fSt16initializer_listI1AE(%3) : ([[INITLIST_TYPE]]) -> ()
