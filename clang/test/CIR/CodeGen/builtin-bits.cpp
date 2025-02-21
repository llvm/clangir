// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir

int test_builtin_clrsb(int x) {
  return __builtin_clrsb(x);
}
// CIR-LABEL: test_builtin_clrsb
// CIR: %{{.+}} = cir.bit.clrsb(%{{.+}} : !s32i) : !s32i

int test_builtin_clrsbl(long x) {
  return __builtin_clrsbl(x);
}
// CIR-LABEL: test_builtin_clrsbl
// CIR: [[tmp:%.+]] = cir.bit.clrsb({{%.+}} : !s64i) : !s64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !s64i), !s32i

int test_builtin_clrsbll(long long x) {
  return __builtin_clrsbll(x);
}
// CIR-LABEL: test_builtin_clrsbll
// CIR: [[tmp:%.+]] = cir.bit.clrsb(%{{.+}} : !s64i) : !s64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !s64i), !s32i

int test_builtin_ctzs(unsigned short x) {
  return __builtin_ctzs(x);
}
// CIR-LABEL: test_builtin_ctzs
// CIR: [[tmp:%.+]] = cir.bit.ctz(%{{.+}} : !u16i) : !u16i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u16i), !s32i

int test_builtin_ctz(unsigned x) {
  return __builtin_ctz(x);
}
// CIR-LABEL: test_builtin_ctz
// CIR: [[tmp:%.+]] = cir.bit.ctz(%{{.+}} : !u32i) : !u32i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u32i), !s32i

int test_builtin_ctzl(unsigned long x) {
  return __builtin_ctzl(x);
}
// CIR-LABEL: test_builtin_ctzl
// CIR: [[tmp:%.+]] = cir.bit.ctz(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_ctzll(unsigned long long x) {
  return __builtin_ctzll(x);
}
// CIR-LABEL: test_builtin_ctzll
// CIR: [[tmp:%.+]] = cir.bit.ctz(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_ctzg(unsigned x) {
  return __builtin_ctzg(x);
}
// CIR-LABEL: test_builtin_ctzg
// CIR: [[tmp:%.+]] = cir.bit.ctz(%{{.+}} : !u32i) : !u32i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u32i), !s32i

int test_builtin_clzs(unsigned short x) {
  return __builtin_clzs(x);
}
// CIR-LABEL: test_builtin_clzs
// CIR: [[tmp:%.+]] = cir.bit.clz(%{{.+}} : !u16i) : !u16i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u16i), !s32i

int test_builtin_clz(unsigned x) {
  return __builtin_clz(x);
}
// CIR-LABEL: cir.func @_Z16test_builtin_clz
// CIR: [[tmp:%.+]] = cir.bit.clz(%{{.+}} : !u32i) : !u32i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u32i), !s32i

int test_builtin_clzl(unsigned long x) {
  return __builtin_clzl(x);
}
// CIR-LABEL: test_builtin_clzl
// CIR: [[tmp:%.+]] = cir.bit.clz(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_clzll(unsigned long long x) {
  return __builtin_clzll(x);
}
// CIR-LABEL: test_builtin_clzll
// CIR: [[tmp:%.+]] = cir.bit.clz(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_clzg(unsigned x) {
  return __builtin_clzg(x);
}
// CIR-LABEL: test_builtin_clz
// CIR: [[tmp:%.+]] = cir.bit.clz(%{{.+}} : !u32i) : !u32i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u32i), !s32i

int test_builtin_ffs(int x) {
  return __builtin_ffs(x);
}
// CIR-LABEL: test_builtin_ffs
// CIR: [[tmp:%.+]] = cir.bit.ffs(%{{.+}} : !s32i) : !s32i

int test_builtin_ffsl(long x) {
  return __builtin_ffsl(x);
}
// CIR-LABEL: test_builtin_ffsl
// CIR: [[tmp:%.+]] = cir.bit.ffs(%{{.+}} : !s64i) : !s64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !s64i), !s32i

int test_builtin_ffsll(long long x) {
  return __builtin_ffsll(x);
}
// CIR-LABEL: test_builtin_ffsll
// CIR: [[tmp:%.+]] = cir.bit.ffs(%{{.+}} : !s64i) : !s64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !s64i), !s32i

int test_builtin_parity(unsigned x) {
  return __builtin_parity(x);
}
// CIR-LABEL: test_builtin_parity
// CIR: [[tmp:%.+]] = cir.bit.parity(%{{.+}} : !u32i) : !u32i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u32i), !s32i

int test_builtin_parityl(unsigned long x) {
  return __builtin_parityl(x);
}
// CIR-LABEL: test_builtin_parityl
// CIR: [[tmp:%.+]] = cir.bit.parity(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_parityll(unsigned long long x) {
  return __builtin_parityll(x);
}
// CIR-LABEL: test_builtin_parityll
// CIR: [[tmp:%.+]] = cir.bit.parity(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_popcount(unsigned x) {
  return __builtin_popcount(x);
}
// CIR-LABEL: test_builtin_popcount
// CIR: [[tmp:%.+]] = cir.bit.popcount(%{{.+}} : !u32i) : !u32i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u32i), !s32i

int test_builtin_popcountl(unsigned long x) {
  return __builtin_popcountl(x);
}
// CIR-LABEL: test_builtin_popcountl
// CIR: [[tmp:%.+]] = cir.bit.popcount(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_popcountll(unsigned long long x) {
  return __builtin_popcountll(x);
}
// CIR-LABEL: test_builtin_popcountll
// CIR: [[tmp:%.+]] = cir.bit.popcount(%{{.+}} : !u64i) : !u64i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u64i), !s32i

int test_builtin_popcountg(unsigned x) {
  return __builtin_popcountg(x);
}
// CIR-LABEL: test_builtin_popcountg
// CIR: [[tmp:%.+]] = cir.bit.popcount(%{{.+}} : !u32i) : !u32i
// CIR: {{%.*}} = cir.cast(integral, [[tmp]] : !u32i), !s32i
