// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-enable -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s


// use output and input operands
// CHECK: cir.func @add1
// CHECK:   [[TMP1:%.*]] = cir.alloca !u32i, cir.ptr <!u32i>, ["x", init] 
// CHECK:   [[TMP2:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["a"]
// CHECK:   [[TMP3:%.*]] = cir.load [[TMP1]] : cir.ptr <!u32i>, !u32i 
// CHECK:   [[TMP4:%.*]] = cir.inline_asm asm_dialect = AD_ATT operand_attrs = [#cir.optnone] "addl $$42, $1", "=r,r,~{dirflag},~{fpsr},~{flags}" [[TMP3]] : (!u32i)
// CHECK:   cir.store [[TMP4]], [[TMP2]] : !s32i, cir.ptr <!s32i> 
unsigned add1(unsigned int x) {  
  int a;
  __asm__("addl $42, %[val]"
      : "=r" (a)
      : [val] "r" (x)
      );
  
  return a;
}

// output operands = input operands
// CHECK: cir.func @add2
// CHECK:   [[TMP1:%.*]] = cir.alloca !u32i, cir.ptr <!u32i>, ["x", init]
// CHECK:   [[TMP2:%.*]] = cir.load [[TMP1]] : cir.ptr <!u32i>, !u32i 
// CHECK:   [[TMP3:%.*]] = cir.inline_asm asm_dialect = AD_ATT operand_attrs = [#cir.optnone] "addl $$42, $0", "=r,0,~{dirflag},~{fpsr},~{flags}" [[TMP2]] : (!u32i) -> !u32i
// CHECK:   cir.store [[TMP3]], [[TMP1]] : !u32i, cir.ptr <!u32i> 
unsigned add2(unsigned int x) {
  __asm__("addl $42, %[val]"
      : [val] "+r" (x)
      );
  return x;
}

// several asm instructions
// CHECK: cir.func @add3
// CHECK:   [[TMP1:%.*]] = cir.alloca !u32i, cir.ptr <!u32i>, ["x", init]
// CHECK:   [[TMP2:%.*]] = cir.load [[TMP1]]
// CHECK:   [[TMP3:%.*]] = cir.inline_asm asm_dialect = AD_ATT operand_attrs = [#cir.optnone] "addl $$42, $0  \0A\09          subl $$1, $0    \0A\09          imul $$2, $0", "=r,0,~{dirflag},~{fpsr},~{flags}" %2 : (!u32i)
// CHECK:   cir.store [[TMP3]], [[TMP1]] : !u32i, cir.ptr <!u32i>
unsigned add3(unsigned int x) {
  __asm__("addl $42, %[val]  \n\t\
          subl $1, %[val]    \n\t\
          imul $2, %[val]"
      : [val] "+r" (x)
      );  
  return x;
}

// use memory operand
// CHECK: cir.func @add4
// CHECK:   [[TMP1:%.*]] = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["x", init] 
// CHECK:   [[TMP2:%.*]] = cir.load deref [[TMP1]] : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i> 
// CHECK:   cir.inline_asm asm_dialect = AD_ATT operand_attrs = [#cir.optnone, {elementtype = !s32i}] "addl $$42, $0", "=*m,~{dirflag},~{fpsr},~{flags}" [[TMP2]] : (!cir.ptr<!s32i>) -> ()
void add4(int *x) {    
  __asm__("addl $42, %[addr]" : [addr] "=m" (*x));
}

/*
There are tests from clang/test/CodeGen/asm.c. No checks for now - we just make
sure no crushes happen
*/

void t1(int len) {
  __asm__ volatile("" : "=&r"(len), "+&r"(len));
}

void t2(unsigned long long t)  {
  __asm__ volatile("" : "+m"(t));
}

void t3(unsigned char *src, unsigned long long temp) {
  __asm__ volatile("" : "+m"(temp), "+r"(src));
}

void t4(void) {
  unsigned long long a;
  struct reg { unsigned long long a, b; } b;

  __asm__ volatile ("":: "m"(a), "m"(b));
}

void t5(int i) {
  asm("nop" : "=r"(i) : "0"(t5));
}

void t6(void) {
  __asm__ volatile("" : : "i" (t6));
}

void t7(int a) {
  __asm__ volatile("T7 NAMED: %[input]" : "+r"(a): [input] "i" (4));  
}

void t8(void) {
  __asm__ volatile("T8 NAMED MODIFIER: %c[input]" :: [input] "i" (4));  
}

unsigned t9(unsigned int a) {
  asm("bswap %0 %1" : "+r" (a));
  return a;
}

void t10(int r) {
  __asm__("PR3908 %[lf] %[xx] %[li] %[r]" : [r] "+r" (r) : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
}

unsigned t11(signed char input) {
  unsigned  output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

unsigned char t12(unsigned input) {
  unsigned char output;
  __asm__("xyz"
          : "=a" (output)
          : "0" (input));
  return output;
}

unsigned char t13(unsigned input) {
  unsigned char output;
  __asm__("xyz %1"
          : "=a" (output)
          : "0" (input));
  return output;
}

struct large {
  int x[1000];
};

unsigned long t15(int x, struct large *P) {
  __asm__("xyz "
          : "=r" (x)
          : "m" (*P), "0" (x));
  return x;
}

// bitfield destination of an asm.
struct S {
  int a : 4;
};

void t14(struct S *P) {
  __asm__("abc %0" : "=r"(P->a) );
}

int t16(void) {
  int a,b;
  asm ( "nop;"
       :"=%c" (a)
       : "r" (b)
       );
  return 0;
}

void t17(void) {
  int i;
  __asm__ ( "nop": "=m"(i));
}

int t18(unsigned data) {
  int a, b;

  asm("xyz" :"=a"(a), "=d"(b) : "a"(data));
  return a + b;
}

int t19(unsigned data) {
  int a, b;

  asm("x{abc|def|ghi}z" :"=r"(a): "r"(data));
  return a + b;
}

// Mismatching source/dest fp types.
double t20(double x) {
  register long double result;
  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;
}

float t21(long double x) {
  register float result;
  __asm __volatile ("frndint"  : "=t" (result) : "0" (x));
  return result;  
}

// accept 'l' constraint
unsigned char t22(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned int bigres;
  unsigned char res;
  __asm__ ("0:\n1:\n" : [bigres] "=la"(bigres) : [la] "0"(la), [lb] "c"(lb) :
                        "edx", "cc");
  res = bigres;
  return res;
}

// accept 'l' constraint
unsigned char t23(unsigned char a, unsigned char b) {
  unsigned int la = a;
  unsigned int lb = b;
  unsigned char res;
  __asm__ ("0:\n1:\n" : [res] "=la"(res) : [la] "0"(la), [lb] "c"(lb) :
                        "edx", "cc");
  return res;
}

void *t24(char c) {
  void *addr;
  __asm__ ("foobar" : "=a" (addr) : "0" (c));
  return addr;
}

void t25(void)
{
  __asm__ __volatile__(					   \
		       "finit"				   \
		       :				   \
		       :				   \
		       :"st","st(1)","st(2)","st(3)",	   \
			"st(4)","st(5)","st(6)","st(7)",   \
			"fpsr","fpcr"			   \
							   );
}

//t26 skipped - no vector type support

// Check to make sure the inline asm non-standard dialect attribute _not_ is
// emitted.
void t27(void) {
  asm volatile("nop");
}

// Check handling of '*' and '#' constraint modifiers.
void t28(void)
{
  asm volatile ("/* %0 */" : : "i#*X,*r" (1));
}

static unsigned t29_var[1];

void t29(void) {
  asm volatile("movl %%eax, %0"
               :
               : "m"(t29_var));
}

void t30(int len) {
  __asm__ volatile(""
                   : "+&&rm"(len));
}

void t31(int len) {
  __asm__ volatile(""
                   : "+%%rm"(len), "+rm"(len));
}

//t32 skipped: no goto

void *t33(void *ptr)
{
  void *ret;
  asm ("lea %1, %0" : "=r" (ret) : "p" (ptr));
  return ret;  
}
