// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

//CHECK: cir.inline_asm has_side_effects asm_dialect = AD_ATT "", ""  : () -> !void
void empty1() {
  __asm__ volatile("" : : : );
}

//CHECK: cir.inline_asm has_side_effects asm_dialect = AD_ATT "xyz", ""  : () -> !void
void empty2() {
  __asm__ volatile("xyz" : : : );
}