// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

template <typename> struct b;
template <typename> struct f;
template <typename c> struct f<c *> {
  typedef c d;
};
template <typename e, typename> class j {
public:
  f<e>::d operator*();
  void operator++();
};
template <typename e, typename g> bool operator!=(j<e, g>, j<e, g>);
template <typename> class k;
template <typename c> struct b<k<c>> {
  using h = c *;
};
template <typename i> struct F {
  typedef b<i>::h h;
  ~F();
};
template <typename c, typename i = k<c>> class G : F<i> {
public:
  typedef j<typename F<i>::h, int> iterator;
  iterator begin();
  iterator end();
};
template <typename l> class m {
public:
  using n = l;
  using o = n *;
  using iterator = o;
  iterator begin();
  iterator end();
};
class p {
public:
  G<p *> u();
  m<p *> r();
} q;

// CHECK: cir.func dso_local @_Z1sv()
// CHECK:   %[[A:.*]] = cir.alloca !rec_m{{.*}}, !cir.ptr<!rec_m{{.*}}>, ["a", init]
// CHECK:   cir.scope {
// CHECK:     cir.for : cond {
// CHECK:     } body {
// CHECK:       cir.scope {
// Verify temporary allocation for range-based for over v->u()
// CHECK:         %{{.*}} = cir.alloca !rec_G{{.*}}, !cir.ptr<!rec_G{{.*}}>, ["ref.tmp0"]
// CHECK:         cir.scope {
// CHECK:           %{{.*}} = cir.call @{{.*}}u{{.*}}(
// CHECK:         }
// CHECK:         cir.for : cond {
// CHECK:         } body {
// CHECK:         } step {
// CHECK:         }
// Verify destructor call for the temporary G object at end of scope
// CHECK:         cir.call @{{.*}}D1Ev(%{{.*}}) : (!cir.ptr<!rec_G{{.*}}>) -> ()
// CHECK:       }
// CHECK:     } step {
// CHECK:     }
// CHECK:   }
// CHECK:   cir.return
void s() {
  m a = q.r();
  for (p *v : a)
    for (p *t : v->u())
      ;
}
