// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

const char16_t *test_utf16() {
  return u"你好世界";
}

//      CHECK: cir.global "private" constant internal @".str" = #cir.const_array<[#cir.int<20320> : !u16i, #cir.int<22909> : !u16i, #cir.int<19990> : !u16i, #cir.int<30028> : !u16i, #cir.int<0> : !u16i]> : !cir.array<!u16i x 5>
// CHECK-NEXT: cir.func @_Z10test_utf16v() -> !cir.ptr<!u16i>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!u16i>, cir.ptr <!cir.ptr<!u16i>>
// CHECK-NEXT:   %1 = cir.get_global @".str" : cir.ptr <!cir.array<!u16i x 5>>
// CHECK-NEXT:   %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!u16i x 5>>), !cir.ptr<!u16i>
// CHECK-NEXT:   cir.store %2, %0 : !cir.ptr<!u16i>, cir.ptr <!cir.ptr<!u16i>>
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.ptr<!u16i>>, !cir.ptr<!u16i>
// CHECK-NEXT:   cir.return %3 : !cir.ptr<!u16i>
// CHECK-NEXT: }

const char32_t *test_utf32() {
  return U"你好世界";
}

//      CHECK: cir.global "private" constant internal @".str1" = #cir.const_array<[#cir.int<20320> : !u32i, #cir.int<22909> : !u32i, #cir.int<19990> : !u32i, #cir.int<30028> : !u32i, #cir.int<0> : !u32i]> : !cir.array<!u32i x 5>
// CHECK-NEXT: cir.func @_Z10test_utf32v() -> !cir.ptr<!u32i>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!u32i>, cir.ptr <!cir.ptr<!u32i>>
// CHECK-NEXT:   %1 = cir.get_global @".str1" : cir.ptr <!cir.array<!u32i x 5>>
// CHECK-NEXT:   %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!u32i x 5>>), !cir.ptr<!u32i>
// CHECK-NEXT:   cir.store %2, %0 : !cir.ptr<!u32i>, cir.ptr <!cir.ptr<!u32i>>
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.ptr<!u32i>>, !cir.ptr<!u32i>
// CHECK-NEXT:   cir.return %3 : !cir.ptr<!u32i>
// CHECK-NEXT: }
