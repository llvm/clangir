// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fenable-clangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void l0() {
  for (;;) {
  }
}

// CHECK: func @_Z2l0v
// CHECK: cir.loop for(cond :  {
// CHECK-NEXT:   %0 = cir.cst(true) : !cir.bool
// CHECK-NEXT:   cir.brcond %0 ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     cir.yield continue
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     cir.yield
// CHECK-NEXT: }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: })  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

void l1() {
  int x = 0;
  for (int i = 0; i < 10; i = i + 1) {
    x = x + 1;
  }
}

// CHECK: func @_Z2l1v
// CHECK: cir.loop for(cond :  {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(10 : i32) : i32
// CHECK-NEXT:   %6 = cir.cmp(lt, %4, %5) : i32, !cir.bool
// CHECK-NEXT:   cir.brcond %6 ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     cir.yield continue
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     cir.yield
// CHECK-NEXT: }, step :  {
// CHECK-NEXT:   %4 = cir.load %2 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %2 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: })  {
// CHECK-NEXT:   %4 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: }

void l2(bool cond) {
  int i = 0;
  while (cond) {
    i = i + 1;
  }
  while (true) {
    i = i + 1;
  }
  while (1) {
    i = i + 1;
  }
}

// CHECK: func @_Z2l2b
// CHECK:         cir.scope {
// CHECK-NEXT:     cir.loop while(cond :  {
// CHECK-NEXT:       %3 = cir.load %0 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:       cir.brcond %3 ^bb1, ^bb2
// CHECK-NEXT:       ^bb1:
// CHECK-NEXT:         cir.yield continue
// CHECK-NEXT:       ^bb2:
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:       cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.loop while(cond :  {
// CHECK-NEXT:       %3 = cir.cst(true) : !cir.bool
// CHECK-NEXT:       cir.brcond %3 ^bb1, ^bb2
// CHECK-NEXT:       ^bb1:
// CHECK-NEXT:         cir.yield continue
// CHECK-NEXT:       ^bb2:
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:       cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.loop while(cond :  {
// CHECK-NEXT:       %3 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %4 = cir.cast(int_to_bool, %3 : i32), !cir.bool
// CHECK-NEXT:       cir.brcond %4 ^bb1, ^bb2
// CHECK-NEXT:       ^bb1:
// CHECK-NEXT:         cir.yield continue
// CHECK-NEXT:       ^bb2:
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:       %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:       %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:       cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }

void l3(bool cond) {
  int i = 0;
  do {
    i = i + 1;
  } while (cond);
  do {
    i = i + 1;
  } while (true);
  do {
    i = i + 1;
  } while (1);
}

// CHECK: func @_Z2l3b
// CHECK: cir.scope {
// CHECK-NEXT:   cir.loop dowhile(cond :  {
// CHECK-NEXT:   %3 = cir.load %0 : cir.ptr <!cir.bool>, !cir.bool
// CHECK-NEXT:   cir.brcond %3 ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     cir.yield continue
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   })  {
// CHECK-NEXT:   %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   cir.loop dowhile(cond :  {
// CHECK-NEXT:   %3 = cir.cst(true) : !cir.bool
// CHECK-NEXT:   cir.brcond %3 ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     cir.yield continue
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   })  {
// CHECK-NEXT:   %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: cir.scope {
// CHECK-NEXT:   cir.loop dowhile(cond :  {
// CHECK-NEXT:   %3 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %4 = cir.cast(int_to_bool, %3 : i32), !cir.bool
// CHECK-NEXT:   cir.brcond %4 ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     cir.yield continue
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     cir.yield
// CHECK-NEXT:   }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   })  {
// CHECK-NEXT:   %3 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:   %4 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %5 = cir.binop(add, %3, %4) : i32
// CHECK-NEXT:   cir.store %5, %1 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.yield
// CHECK-NEXT:   }
// CHECK-NEXT: }

void l4() {
  int i = 0, y = 100;
  while (true) {
    i = i + 1;
    if (i < 10)
      continue;
    y = y - 20;
  }
}

// CHECK: func @_Z2l4v
// CHECK: cir.loop while(cond :  {
// CHECK-NEXT:   %4 = cir.cst(true) : !cir.bool
// CHECK-NEXT:   cir.brcond %4 ^bb1, ^bb2
// CHECK-NEXT:   ^bb1:
// CHECK-NEXT:     cir.yield continue
// CHECK-NEXT:   ^bb2:
// CHECK-NEXT:     cir.yield
// CHECK-NEXT: }, step :  {
// CHECK-NEXT:   cir.yield
// CHECK-NEXT: })  {
// CHECK-NEXT:   %4 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:   %5 = cir.cst(1 : i32) : i32
// CHECK-NEXT:   %6 = cir.binop(add, %4, %5) : i32
// CHECK-NEXT:   cir.store %6, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     %10 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:     %11 = cir.cst(10 : i32) : i32
// CHECK-NEXT:     %12 = cir.cmp(lt, %10, %11) : i32, !cir.bool
// CHECK-NEXT:     cir.if %12 {
// CHECK-NEXT:       cir.yield continue
// CHECK-NEXT:     }
// CHECK-NEXT:   }

void l5() {
  do {
  } while (0);
}

// CHECK: func @_Z2l5v() {
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.loop dowhile(cond :  {
// CHECK-NEXT:       %0 = cir.cst(0 : i32) : i32
// CHECK-NEXT:       %1 = cir.cast(int_to_bool, %0 : i32), !cir.bool
// CHECK-NEXT:       cir.brcond %1 ^bb1, ^bb2
// CHECK-NEXT:       ^bb1:
// CHECK-NEXT:         cir.yield continue
// CHECK-NEXT:       ^bb2:
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }

void l6() {
  while (true) {
    return;
  }
}

// CHECK: func @_Z2l6v() {
// CHECK-NEXT:   cir.scope {
// CHECK-NEXT:     cir.loop while(cond :  {
// CHECK-NEXT:       %0 = cir.cst(true) : !cir.bool
// CHECK-NEXT:       cir.brcond %0 ^bb1, ^bb2
// CHECK-NEXT:       ^bb1:
// CHECK-NEXT:         cir.yield continue
// CHECK-NEXT:       ^bb2:
// CHECK-NEXT:         cir.yield
// CHECK-NEXT:     }, step :  {
// CHECK-NEXT:       cir.yield
// CHECK-NEXT:     })  {
// CHECK-NEXT:       cir.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   cir.return
// CHECK-NEXT: }
