struct IncompleteType;

void foo() {
  IncompleteType *f;
  f->x;
}
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):6 %s -o - | FileCheck %s -allow-empty
// CHECK-NOT: COMPLETION:
