// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir-analysis-only -fclangir-lifetime-check="history=all;remarks=all;history_limit=1" -clangir-verify-diagnostics -emit-obj %s -o /dev/null

struct [[gsl::Owner(int)]] MyIntOwner {
  int val;
  MyIntOwner(int v) : val(v) {}
  void changeInt(int i);
  int &operator*();
  int read() const;
};

struct [[gsl::Pointer(int)]] MyIntPointer {
  int *ptr;
  MyIntPointer(int *p = nullptr) : ptr(p) {}
  MyIntPointer(const MyIntOwner &);
  int &operator*();
  MyIntOwner toOwner();
  int read() { return *ptr; }
};

void yolo() {
  MyIntPointer p;
  {
    MyIntOwner o(1);
    p = o;
    *p = 3; // expected-remark {{pset => { o__1' }}}
  }       // expected-note {{pointee 'o' invalidated at end of scope}}
  *p = 4; // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
}

void yolo2() {
  MyIntPointer p;
  MyIntOwner o(1);
  p = o;
  (void)o.read();
  (void)p.read(); // expected-remark {{pset => { o__1' }}}
  o.changeInt(42); // expected-note {{invalidated by non-const use of owner type}}
  (void)p.read(); // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
  p = o;
  (void)p.read(); // expected-remark {{pset => { o__2' }}}
  o.changeInt(33); // expected-note {{invalidated by non-const use of owner type}}
  (void)p.read(); // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
}

void yolo3() {
  MyIntPointer p, q;
  MyIntOwner o(1);
  p = o;
  q = o;
  (void)q.read(); // expected-remark {{pset => { o__1' }}}
  (void)p.read(); // expected-remark {{pset => { o__1' }}}
  o.changeInt(42); // expected-note {{invalidated by non-const use of owner type}}
  (void)p.read(); // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
  (void)q.read(); // expected-warning {{use of invalid pointer 'q'}}
  // expected-remark@-1 {{pset => { invalid }}}
}

void yolo4() {
  MyIntOwner o0(1);
  MyIntOwner o1(2);
  MyIntPointer p{o0}, q{o1};
  p.read(); // expected-remark {{pset => { o0__1' }}}
  q.read(); // expected-remark {{pset => { o1__1' }}}
  o0 = o1; // expected-note {{invalidated by non-const use of owner type}}
  p.read(); // expected-warning {{use of invalid pointer 'p'}}
  // expected-remark@-1 {{pset => { invalid }}}
  q.read(); // expected-remark {{pset => { o1__1' }}}
}