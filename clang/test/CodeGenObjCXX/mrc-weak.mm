// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.10 -emit-llvm -fblocks -fobjc-weak -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-MODERN
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.10 -emit-llvm -fblocks -fobjc-weak -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-FRAGILE

@interface Object
- (instancetype) retain;
- (void) run;
@end

// CHECK-MODERN: @OBJC_CLASS_NAME_{{.*}} = {{.*}} c"\01\00"
// CHECK-MODERN: @"_OBJC_CLASS_RO_$_Foo" = {{.*}} { i32 772
//   772 == 0x304
//            ^ HasMRCWeakIvars
//            ^ HasCXXDestructorOnly
//              ^ HasCXXStructors

// CHECK-FRAGILE: @OBJC_CLASS_NAME_{{.*}} = {{.*}} c"\01\00"
// CHECK-FRAGILE: @OBJC_CLASS_Foo = {{.*}} i32 134225921,
//   134225921 == 0x08002001
//                   ^ HasMRCWeakIvars
//                      ^ HasCXXStructors
//                         ^ Factory
@interface Foo : Object {
  __weak id ivar;
}
@end

@implementation Foo
// CHECK-LABEL: define internal void @"\01-[Foo .cxx_destruct]"
// CHECK: call void @llvm.objc.destroyWeak
@end


void test1(__weak id x) {}
// CHECK-LABEL: define{{.*}} void @_Z5test1P11objc_object(
// CHECK:      [[X:%.*]] = alloca ptr,
// CHECK-NEXT: llvm.objc.initWeak
// CHECK-NEXT: llvm.objc.destroyWeak
// CHECK-NEXT: ret void

void test2(id y) {
  __weak id z = y;
}
// CHECK-LABEL: define{{.*}} void @_Z5test2P11objc_object(
// CHECK:      [[Y:%.*]] = alloca ptr,
// CHECK-NEXT: [[Z:%.*]] = alloca ptr,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
// CHECK-NEXT: call ptr @llvm.objc.initWeak(ptr [[Z]], ptr [[T0]])
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[Z]])
// CHECK-NEXT: ret void

void test3(id y) {
  __weak id z;
  z = y;
}
// CHECK-LABEL: define{{.*}} void @_Z5test3P11objc_object(
// CHECK:      [[Y:%.*]] = alloca ptr,
// CHECK-NEXT: [[Z:%.*]] = alloca ptr,
// CHECK-NEXT: store
// CHECK-NEXT: store ptr null, ptr [[Z]]
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[Y]]
// CHECK-NEXT: call ptr @llvm.objc.storeWeak(ptr [[Z]], ptr [[T0]])
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr [[Z]])
// CHECK-NEXT: ret void

void test4(__weak id *p) {
  id y = *p;
}
// CHECK-LABEL: define{{.*}} void @_Z5test4PU6__weakP11objc_object(
// CHECK:      [[P:%.*]] = alloca ptr,
// CHECK-NEXT: [[Y:%.*]] = alloca ptr,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[P]]
// CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.loadWeak(ptr [[T0]])
// CHECK-NEXT: store ptr [[T1]], ptr [[Y]]
// CHECK-NEXT: ret void

void test5(__weak id *p) {
  id y = [*p retain];
}
// CHECK-LABEL: define{{.*}} void @_Z5test5PU6__weakP11objc_object
// CHECK:      [[P:%.*]] = alloca ptr,
// CHECK-NEXT: [[Y:%.*]] = alloca ptr,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[P]]
// CHECK-NEXT: [[T1:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[T0]])
// CHECK-NEXT: store ptr [[T1]], ptr [[Y]]
// CHECK-NEXT: ret void

void test6(__weak Foo **p) {
  Foo *y = [*p retain];
}
// CHECK-LABEL: define{{.*}} void @_Z5test6PU6__weakP3Foo
// CHECK:      [[P:%.*]] = alloca ptr,
// CHECK-NEXT: [[Y:%.*]] = alloca ptr,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[P]]
// CHECK-NEXT: [[T2:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[T0]])
// CHECK-NEXT: store ptr [[T2]], ptr [[Y]]
// CHECK-NEXT: ret void

extern "C" id get_object(void);
extern "C" void use_block(void (^)(void));

void test7(void) {
  __weak Foo *p = get_object();
  use_block(^{ [p run ]; });
}
// CHECK-LABEL: define{{.*}} void @_Z5test7v
// CHECK:       [[P:%.*]] = alloca ptr,
// CHECK:       [[T0:%.*]] = call ptr @get_object()
// CHECK-NEXT:  call ptr @llvm.objc.initWeak(ptr [[P]], ptr [[T0]])
// CHECK:       call void @llvm.objc.copyWeak
// CHECK:       call void @use_block
// CHECK:       call void @llvm.objc.destroyWeak

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block
// CHECK:       @llvm.objc.copyWeak

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block
// CHECK:       @llvm.objc.destroyWeak

void test8(void) {
  __block __weak Foo *p = get_object();
  use_block(^{ [p run ]; });
}
// CHECK-LABEL: define{{.*}} void @_Z5test8v
// CHECK:       call ptr @llvm.objc.initWeak
// CHECK-NOT:   call void @llvm.objc.copyWeak
// CHECK:       call void @use_block
// CHECK:       call void @llvm.objc.destroyWeak

// CHECK-LABEL: define internal void @__Block_byref_object_copy
// CHECK:       call void @llvm.objc.moveWeak

// CHECK-LABEL: define internal void @__Block_byref_object_dispose
// CHECK:       call void @llvm.objc.destroyWeak

// CHECK-LABEL: define{{.*}} void @_Z14test9_baselinev()
// CHECK:       define linkonce_odr hidden void @__copy_helper
// CHECK:       define linkonce_odr hidden void @__destroy_helper
void test9_baseline(void) {
  Foo *p = get_object();
  use_block(^{ [p run]; });
}

// CHECK-LABEL: define{{.*}} void @_Z5test9v()
// CHECK-NOT:   define internal void @__copy_helper
// CHECK-NOT:   define internal void @__destroy_helper
// CHECK:       define{{.*}} void @_Z9test9_finv()
void test9(void) {
  __unsafe_unretained Foo *p = get_object();
  use_block(^{ [p run]; });
}
void test9_fin() {}

// CHECK-LABEL: define{{.*}} void @_Z6test10v()
// CHECK-NOT:   define internal void @__copy_helper
// CHECK-NOT:   define internal void @__destroy_helper
// CHECK:       define{{.*}} void @_Z10test10_finv()
void test10(void) {
  typedef __unsafe_unretained Foo *UnsafeFooPtr;
  UnsafeFooPtr p = get_object();
  use_block(^{ [p run]; });
}
void test10_fin() {}

// CHECK-LABEL: define weak_odr void @_Z6test11ILj0EEvv()
// CHECK-NOT:   define internal void @__copy_helper
// CHECK-NOT:   define internal void @__destroy_helper
// CHECK:       define{{.*}} void @_Z10test11_finv()
template <unsigned i> void test11(void) {
  typedef __unsafe_unretained Foo *UnsafeFooPtr;
  UnsafeFooPtr p = get_object();
  use_block(^{ [p run]; });
}
template void test11<0>();
void test11_fin() {}
