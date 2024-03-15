// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

//CHECK: cir.asm(x86_att, {"" "~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone] side_effects  : () -> ()
void empty1() {
  __asm__ volatile("" : : : );
}

//CHECK: cir.asm(x86_att, {"xyz" "~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone] side_effects  : () -> ()
void empty2() {
  __asm__ volatile("xyz" : : : );
}

//CHECK: cir.asm(x86_att, {"" "=*m,*m,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone, !s32i, !s32i] side_effects %0, %0 : (!cir.ptr<!s32i>, !cir.ptr<!s32i>) -> ()
void t1(int x) {
  __asm__ volatile("" : "+m"(x));
}

//CHECK: cir.asm(x86_att, {"" "*m,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone, !s32i] side_effects %0 : (!cir.ptr<!s32i>) -> ()
void t2(int x) {
  __asm__ volatile("" : : "m"(x));
}

//CHECK: cir.asm(x86_att, {"" "=*m,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone, !s32i] side_effects %0 : (!cir.ptr<!s32i>) -> ()
void t3(int x) {
  __asm__ volatile("" : "=m"(x));
}

//CHECK: cir.asm(x86_att, {"" "=&r,=&r,1,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone] side_effects %2 : (!s32i) -> !ty_22anon2E022
void t4(int x) {
  __asm__ volatile("" : "=&r"(x), "+&r"(x));
}

// CHECK: {{.*}} = cir.asm(x86_att, {"addl $$42, $1" "=r,r,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone] {{.*}} : (!u32i) -> !s32i
unsigned add1(unsigned int x) {  
  int a;
  __asm__("addl $42, %[val]"
      : "=r" (a)
      : [val] "r" (x)
      );
  
  return a;
}

// CHECK: {{.*}} = cir.asm(x86_att, {"addl $$42, $0" "=r,0,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone] {{.*}} : (!u32i) -> !u32i
unsigned add2(unsigned int x) {
  __asm__("addl $42, %[val]"
      : [val] "+r" (x)
      );
  return x;
}

// CHECK: {{.*}} = cir.asm(x86_att, {"addl $$42, $0  \0A\09          subl $$1, $0    \0A\09          imul $$2, $0" "=r,0,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone] {{.*}} : (!u32i) -> !u32i
unsigned add3(unsigned int x) { // ((42 + x) - 1) * 2
  __asm__("addl $42, %[val]  \n\t\
          subl $1, %[val]    \n\t\
          imul $2, %[val]"
      : [val] "+r" (x)
      );  
  return x;
}

// CHECK: [[TMP:%.*]] = cir.load deref {{.*}} : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: cir.asm(x86_att, {"addl $$42, $0" "=*m,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone, !s32i] [[TMP]] : (!cir.ptr<!s32i>) -> ()
void add4(int *x) {    
  __asm__("addl $42, %[addr]" : [addr] "=m" (*x));
}

// CHECK: {{.*}} = cir.asm(x86_att, {"fadd $0, $1" "=&{st},f,~{dirflag},~{fpsr},~{flags}"}) operand_attrs = [#cir.optnone] {{.*}} : (!cir.float) -> !cir.float
float add5(float x, float y) {
  __asm__("fadd %[x], %[y]"
      : [x] "=&t" (x)
      : [y] "f" (y)
      );
  return x;
}