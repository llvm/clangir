struct Trivial{ int i; };
void f(Trivial a){
    Trivial b(a);
    b=a;
}