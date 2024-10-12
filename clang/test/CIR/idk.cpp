class a {
};
struct b;
template <typename> class ref {};
class c {
  a mA(b *);
  ref<a> mB(b *, long);
};
typedef struct d *tdA;


tdA fnA(b *, a, char *, char *);

ref<a> c::mB(b *arg, long) {
  static tdA val = fnA(arg, mA(arg), "", "");
}

