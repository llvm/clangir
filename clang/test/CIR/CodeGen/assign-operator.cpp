// RUN: %clang_cc1 -std=c++17 -mconstructor-aliases -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int strlen(char const *);

struct String {
  long size;
  long capacity;

  String() : size{0}, capacity{0} {}
  String(char const *s) : size{strlen(s)}, capacity{size} {}
  // StringView::StringView(String const&)
  //
  // CHECK: cir.func linkonce_odr @_ZN10StringViewC2ERK6String
  // CHECK:   %0 = cir.alloca !cir.ptr<!rec_StringView>, !cir.ptr<!cir.ptr<!rec_StringView>>, ["this", init] {alignment = 8 : i64}
  // CHECK:   %1 = cir.alloca !cir.ptr<!rec_String>, !cir.ptr<!cir.ptr<!rec_String>>, ["s", init, const] {alignment = 8 : i64}
  // CHECK:   cir.store %arg0, %0 : !cir.ptr<!rec_StringView>
  // CHECK:   cir.store %arg1, %1 : !cir.ptr<!rec_String>
  // CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_StringView>>

  // Get address of `this->size`

  // CHECK:   %3 = cir.get_member %2[0] {name = "size"}

  // Get address of `s`

  // CHECK:   %4 = cir.load %1 : !cir.ptr<!cir.ptr<!rec_String>>

  // Get the address of s.size

  // CHECK:   %5 = cir.get_member %4[0] {name = "size"}

  // Load value from s.size and store in this->size

  // CHECK:   %6 = cir.load %5 : !cir.ptr<!s64i>, !s64i
  // CHECK:   cir.store %6, %3 : !s64i, !cir.ptr<!s64i>
  // CHECK:   cir.return
  // CHECK: }

  // StringView::operator=(StringView&&)
  //
  // CHECK: cir.func linkonce_odr @_ZN10StringViewaSEOS_
  // CHECK:   %0 = cir.alloca !cir.ptr<!rec_StringView>, !cir.ptr<!cir.ptr<!rec_StringView>>, ["this", init] {alignment = 8 : i64}
  // CHECK:   %1 = cir.alloca !cir.ptr<!rec_StringView>, !cir.ptr<!cir.ptr<!rec_StringView>>, ["", init, const] {alignment = 8 : i64}
  // CHECK:   %2 = cir.alloca !cir.ptr<!rec_StringView>, !cir.ptr<!cir.ptr<!rec_StringView>>, ["__retval"] {alignment = 8 : i64}
  // CHECK:   cir.store %arg0, %0 : !cir.ptr<!rec_StringView>
  // CHECK:   cir.store %arg1, %1 : !cir.ptr<!rec_StringView>
  // CHECK:   %3 = cir.load deref %0 : !cir.ptr<!cir.ptr<!rec_StringView>>
  // CHECK:   %4 = cir.load %1 : !cir.ptr<!cir.ptr<!rec_StringView>>
  // CHECK:   %5 = cir.get_member %4[0] {name = "size"}
  // CHECK:   %6 = cir.load %5 : !cir.ptr<!s64i>, !s64i
  // CHECK:   %7 = cir.get_member %3[0] {name = "size"}
  // CHECK:   cir.store %6, %7 : !s64i, !cir.ptr<!s64i>
  // CHECK:   cir.store %3, %2 : !cir.ptr<!rec_StringView>
  // CHECK:   %8 = cir.load %2 : !cir.ptr<!cir.ptr<!rec_StringView>>
  // CHECK:   cir.return %8 : !cir.ptr<!rec_StringView>
  // CHECK: }
};

struct StringView {
  long size;

  StringView(const String &s) : size{s.size} {}
  StringView() : size{0} {}
};

int main() {
  StringView sv;
  {
    String s = "Hi";
    sv = s;
  }
}

// CHECK: cir.func @main() -> !s32i
// CHECK:     %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK:     %1 = cir.alloca !rec_StringView, !cir.ptr<!rec_StringView>, ["sv", init] {alignment = 8 : i64}
// CHECK:     cir.call @_ZN10StringViewC2Ev(%1) : (!cir.ptr<!rec_StringView>) -> ()
// CHECK:     cir.scope {
// CHECK:       %3 = cir.alloca !rec_String, !cir.ptr<!rec_String>, ["s", init] {alignment = 8 : i64}
// CHECK:       %4 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>>
// CHECK:       %5 = cir.cast(array_to_ptrdecay, %4 : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// CHECK:       cir.call @_ZN6StringC2EPKc(%3, %5) : (!cir.ptr<!rec_String>, !cir.ptr<!s8i>) -> ()
// CHECK:       cir.scope {
// CHECK:         %6 = cir.alloca !rec_StringView, !cir.ptr<!rec_StringView>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:         cir.call @_ZN10StringViewC2ERK6String(%6, %3) : (!cir.ptr<!rec_StringView>, !cir.ptr<!rec_String>) -> ()
// CHECK:         %7 = cir.call @_ZN10StringViewaSEOS_(%1, %6) : (!cir.ptr<!rec_StringView>, !cir.ptr<!rec_StringView>) -> !cir.ptr<!rec_StringView>
// CHECK:       }
// CHECK:     }
// CHECK:     %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK:     cir.return %2 : !s32i
// CHECK: }

struct HasNonTrivialAssignOp {
  HasNonTrivialAssignOp &operator=(const HasNonTrivialAssignOp &);
};

struct ContainsNonTrivial {
  HasNonTrivialAssignOp start;
  int i;
  int *j;
  HasNonTrivialAssignOp middle;
  int k : 4;
  int l : 4;
  int m : 4;
  HasNonTrivialAssignOp end;
  ContainsNonTrivial &operator=(const ContainsNonTrivial &);
};

// CHECK-LABEL: cir.func @_ZN18ContainsNonTrivialaSERKS_(
// CHECK-NEXT:    %[[#THIS:]] = cir.alloca !cir.ptr<!rec_ContainsNonTrivial>
// CHECK-NEXT:    %[[#OTHER:]] = cir.alloca !cir.ptr<!rec_ContainsNonTrivial>
// CHECK-NEXT:    %[[#RETVAL:]] = cir.alloca !cir.ptr<!rec_ContainsNonTrivial>
// CHECK-NEXT:    cir.store %arg0, %[[#THIS]]
// CHECK-NEXT:    cir.store %arg1, %[[#OTHER]]
// CHECK-NEXT:    %[[#THIS_LOAD:]] = cir.load deref %[[#THIS]]
// CHECK-NEXT:    %[[#THIS_START:]] = cir.get_member %[[#THIS_LOAD]][0] {name = "start"}
// CHECK-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CHECK-NEXT:    %[[#OTHER_START:]] = cir.get_member %[[#OTHER_LOAD]][0] {name = "start"}
// CHECK-NEXT:    cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(%[[#THIS_START]], %[[#OTHER_START]])
// CHECK-NEXT:    %[[#THIS_I:]] = cir.get_member %[[#THIS_LOAD]][2] {name = "i"}
// CHECK-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CHECK-NEXT:    %[[#OTHER_I:]] = cir.get_member %[[#OTHER_LOAD]][2] {name = "i"}
// CHECK-NEXT:    %[[#MEMCPY_SIZE:]] = cir.const #cir.int<12> : !u64i
// CHECK-NEXT:    %[[#THIS_I_CAST:]] = cir.cast(bitcast, %[[#THIS_I]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CHECK-NEXT:    %[[#OTHER_I_CAST:]] = cir.cast(bitcast, %[[#OTHER_I]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CHECK-NEXT:    cir.libc.memcpy %[[#MEMCPY_SIZE]] bytes from %[[#OTHER_I_CAST]] to %[[#THIS_I_CAST]]
// CHECK-NEXT:    %[[#THIS_MIDDLE:]] = cir.get_member %[[#THIS_LOAD]][4] {name = "middle"}
// CHECK-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CHECK-NEXT:    %[[#OTHER_MIDDLE:]] = cir.get_member %[[#OTHER_LOAD]][4] {name = "middle"}
// CHECK-NEXT:    cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(%[[#THIS_MIDDLE]], %[[#OTHER_MIDDLE]])
// CHECK-NEXT:    %[[#THIS_K:]] = cir.get_member %[[#THIS_LOAD]][5] {name = "k"}
// CHECK-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CHECK-NEXT:    %[[#OTHER_K:]] = cir.get_member %[[#OTHER_LOAD]][5] {name = "k"}
// CHECK-NEXT:    %[[#MEMCPY_SIZE:]] = cir.const #cir.int<2> : !u64i
// CHECK-NEXT:    %[[#THIS_K_CAST:]] = cir.cast(bitcast, %[[#THIS_K]] : !cir.ptr<!u16i>), !cir.ptr<!void>
// CHECK-NEXT:    %[[#OTHER_K_CAST:]] = cir.cast(bitcast, %[[#OTHER_K]] : !cir.ptr<!u16i>), !cir.ptr<!void>
// CHECK-NEXT:    cir.libc.memcpy %[[#MEMCPY_SIZE]] bytes from %[[#OTHER_K_CAST]] to %[[#THIS_K_CAST]]
// CHECK-NEXT:    %[[#THIS_END:]] = cir.get_member %[[#THIS_LOAD]][6] {name = "end"}
// CHECK-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CHECK-NEXT:    %[[#OTHER_END:]] = cir.get_member %[[#OTHER_LOAD]][6] {name = "end"}
// CHECK-NEXT:    cir.call @_ZN21HasNonTrivialAssignOpaSERKS_(%[[#THIS_END]], %[[#OTHER_END]])
// CHECK-NEXT:    cir.store %[[#THIS_LOAD]], %[[#RETVAL]]
// CHECK-NEXT:    %[[#RETVAL_LOAD:]] = cir.load %[[#RETVAL]]
// CHECK-NEXT:    cir.return %[[#RETVAL_LOAD]]
// CHECK-NEXT:  }
ContainsNonTrivial &
ContainsNonTrivial::operator=(const ContainsNonTrivial &) = default;

struct Trivial {
  int i;
  int *j;
  double k;
  int l[3];
};

// CHECK-LABEL: cir.func linkonce_odr @_ZN7TrivialaSERKS_(
// CHECK-NEXT:    %[[#THIS:]] = cir.alloca !cir.ptr<!rec_Trivial>
// CHECK-NEXT:    %[[#OTHER:]] = cir.alloca !cir.ptr<!rec_Trivial>
// CHECK-NEXT:    %[[#RETVAL:]] = cir.alloca !cir.ptr<!rec_Trivial>
// CHECK-NEXT:    cir.store %arg0, %[[#THIS]]
// CHECK-NEXT:    cir.store %arg1, %[[#OTHER]]
// CHECK-NEXT:    %[[#THIS_LOAD:]] = cir.load deref %[[#THIS]]
// CHECK-NEXT:    %[[#THIS_I:]] = cir.get_member %[[#THIS_LOAD]][0] {name = "i"}
// CHECK-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#OTHER]]
// CHECK-NEXT:    %[[#OTHER_I:]] = cir.get_member %[[#OTHER_LOAD]][0] {name = "i"}
// Note that tail padding bytes are not included.
// CHECK-NEXT:    %[[#MEMCPY_SIZE:]] = cir.const #cir.int<36> : !u64i
// CHECK-NEXT:    %[[#THIS_I_CAST:]] = cir.cast(bitcast, %[[#THIS_I]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CHECK-NEXT:    %[[#OTHER_I_CAST:]] = cir.cast(bitcast, %[[#OTHER_I]] : !cir.ptr<!s32i>), !cir.ptr<!void>
// CHECK-NEXT:    cir.libc.memcpy %[[#MEMCPY_SIZE]] bytes from %[[#OTHER_I_CAST]] to %[[#THIS_I_CAST]]
// CHECK-NEXT:    cir.store %[[#THIS_LOAD]], %[[#RETVAL]]
// CHECK-NEXT:    %[[#RETVAL_LOAD:]] = cir.load %[[#RETVAL]]
// CHECK-NEXT:    cir.return %[[#RETVAL_LOAD]]
// CHECK-NEXT:  }

// We should explicitly call operator= even for trivial types.
// CHECK-LABEL: cir.func @_Z11copyTrivialR7TrivialS0_(
// CHECK:         cir.call @_ZN7TrivialaSERKS_(
void copyTrivial(Trivial &a, Trivial &b) {
  a = b;
}

struct ContainsTrivial {
  Trivial t1;
  Trivial t2;
  ContainsTrivial &operator=(const ContainsTrivial &);
};

// We should explicitly call operator= even for trivial types.
// CHECK-LABEL: cir.func @_ZN15ContainsTrivialaSERKS_(
// CHECK:         cir.call @_ZN7TrivialaSERKS_(
// CHECK:         cir.call @_ZN7TrivialaSERKS_(
ContainsTrivial &ContainsTrivial::operator=(const ContainsTrivial &) = default;

struct ContainsTrivialArray {
  Trivial arr[2];
  ContainsTrivialArray &operator=(const ContainsTrivialArray &);
};

// We should be calling operator= here but don't currently.
// CHECK-LABEL: cir.func @_ZN20ContainsTrivialArrayaSERKS_(
// CHECK:         %[[#THIS_LOAD:]] = cir.load deref %[[#]]
// CHECK-NEXT:    %[[#THIS_ARR:]] = cir.get_member %[[#THIS_LOAD]][0] {name = "arr"}
// CHECK-NEXT:    %[[#THIS_ARR_CAST:]] = cir.cast(bitcast, %[[#THIS_ARR]] : !cir.ptr<!cir.array<!rec_Trivial x 2>>), !cir.ptr<!void>
// CHECK-NEXT:    %[[#OTHER_LOAD:]] = cir.load %[[#]]
// CHECK-NEXT:    %[[#OTHER_ARR:]] = cir.get_member %[[#OTHER_LOAD]][0] {name = "arr"}
// CHECK-NEXT:    %[[#OTHER_ARR_CAST:]] = cir.cast(bitcast, %[[#OTHER_ARR]] : !cir.ptr<!cir.array<!rec_Trivial x 2>>), !cir.ptr<!void>
// CHECK-NEXT:    %[[#MEMCPY_SIZE:]] = cir.const #cir.int<80> : !u64i
// CHECK-NEXT:    cir.libc.memcpy %[[#MEMCPY_SIZE]] bytes from %[[#OTHER_ARR_CAST]] to %[[#THIS_ARR_CAST]]
ContainsTrivialArray &
ContainsTrivialArray::operator=(const ContainsTrivialArray &) = default;
