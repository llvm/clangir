// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Base {
  virtual ~Base();
};
// CHECK: !ty_22Base22 = !cir.struct

struct Derived : Base {};
// CHECK: !ty_22Derived22 = !cir.struct

Derived *ptr_cast(Base *b) {
  return dynamic_cast<Derived *>(b);
}

// CHECK: cir.func @_Z8ptr_castP4Base
// CHECK:   %{{.+}} = cir.dyn_cast(%{{.+}} : !cir.ptr<!ty_22Base22>, #cir.dyn_cast_info<#cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>, #cir.downcast_info<0, false>>) -> !cir.ptr<!ty_22Derived22>
// CHECK: }

Derived &ref_cast(Base &b) {
  return dynamic_cast<Derived &>(b);
}

// CHECK: cir.func @_Z8ref_castR4Base
// CHECK:   %{{.+}} = cir.dyn_cast(%{{.+}} : !cir.ptr<!ty_22Base22>, refcast, #cir.dyn_cast_info<#cir.global_view<@_ZTI4Base> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI7Derived> : !cir.ptr<!u8i>, #cir.downcast_info<0, false>>) -> !cir.ptr<!ty_22Derived22>
// CHECK: }

void *ptr_cast_to_complete(Base *ptr) {
  return dynamic_cast<void *>(ptr);
}

//      CHECK: cir.func @_Z20ptr_cast_to_completeP4Base
//      CHECK:   %[[#V19:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!ty_22Base22>>, !cir.ptr<!ty_22Base22>
// CHECK-NEXT:   %[[#V20:]] = cir.cast(ptr_to_bool, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.bool
// CHECK-NEXT:   %[[#V21:]] = cir.unary(not, %[[#V20]]) : !cir.bool, !cir.bool
// CHECK-NEXT:   %{{.+}} = cir.ternary(%[[#V21]], true {
// CHECK-NEXT:     %[[#V22:]] = cir.const(#cir.ptr<null> : !cir.ptr<!void>) : !cir.ptr<!void>
// CHECK-NEXT:     cir.yield %[[#V22]] : !cir.ptr<!void>
// CHECK-NEXT:   }, false {
// CHECK-NEXT:     %[[#V23:]] = cir.cast(bitcast, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!cir.ptr<!s64i>>
// CHECK-NEXT:     %[[#V24:]] = cir.load %[[#V23]] : cir.ptr <!cir.ptr<!s64i>>, !cir.ptr<!s64i>
// CHECK-NEXT:     %[[#V25:]] = cir.vtable.address_point( %[[#V24]] : !cir.ptr<!s64i>, vtable_index = 0, address_point_index = -2) : cir.ptr <!s64i>
// CHECK-NEXT:     %[[#V26:]] = cir.load %[[#V25]] : cir.ptr <!s64i>, !s64i
// CHECK-NEXT:     %[[#V27:]] = cir.cast(bitcast, %[[#V19]] : !cir.ptr<!ty_22Base22>), !cir.ptr<!u8i>
// CHECK-NEXT:     %[[#V28:]] = cir.ptr_stride(%[[#V27]] : !cir.ptr<!u8i>, %[[#V26]] : !s64i), !cir.ptr<!u8i>
// CHECK-NEXT:     %[[#V29:]] = cir.cast(bitcast, %[[#V28]] : !cir.ptr<!u8i>), !cir.ptr<!void>
// CHECK-NEXT:     cir.yield %[[#V29]] : !cir.ptr<!void>
// CHECK-NEXT:   }) : (!cir.bool) -> !cir.ptr<!void>
