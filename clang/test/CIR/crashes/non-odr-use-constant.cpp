// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Non-ODR-use constant expression not implemented
// Location: CIRGenExpr.cpp:1122

namespace llvm {
  template<typename ValueTy> class StringMapEntry {};
  template<typename ValueTy> class StringMapIterBase {
  public:
    StringMapEntry<ValueTy>& operator*() const;
    StringMapIterBase& operator++();
    friend bool operator!=(const StringMapIterBase& LHS, const StringMapIterBase& RHS);
  };
  template<typename ValueTy> class StringMap {
  public:
    StringMapIterBase<ValueTy> begin();
    StringMapIterBase<ValueTy> end();
  };
  struct EmptyStringSetTag {};
  template<class AllocatorTy = int> class StringSet : public StringMap<EmptyStringSetTag> {};
}

namespace clang {
  static llvm::StringSet<> BuiltinClasses;

  void EmitBuiltins() {
    for (const auto &Entry : BuiltinClasses) {
    }
  }
}
