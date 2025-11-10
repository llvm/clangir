// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: !rec_Zero = !cir.record<struct "Zero" padded {!u8i}>

struct Zero {
  void yolo();
};

void f() {
  Zero z0 = Zero();
  // {} no element init.
  Zero z1 = Zero{};
}

// TODO: In this case, z1 gets "initialized" with an undef value. Should we
//       treat that as uninitialized? Should it even be happening?

// CHECK: cir.func dso_local @_Z1fv()
// CHECK:     %[[Z0:.*]] = cir.alloca !rec_Zero, !cir.ptr<!rec_Zero>, ["z0", init]
// CHECK:     %[[Z1:.*]] = cir.alloca !rec_Zero, !cir.ptr<!rec_Zero>, ["z1", init]
// CHECK:     cir.call @_ZN4ZeroC1Ev(%[[Z0]]) : (!cir.ptr<!rec_Zero>) -> ()
// CHECK:     %[[UNDEF:.*]] = cir.const #cir.undef : !rec_Zero
// CHECK:     cir.store{{.*}} %[[UNDEF]], %[[Z1]] : !rec_Zero, !cir.ptr<!rec_Zero>
// CHECK:     cir.return
