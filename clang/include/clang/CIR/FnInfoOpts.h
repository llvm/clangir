#ifndef CIR_FNINFOOPTS_H
#define CIR_FNINFOOPTS_H

#include "llvm/ADT/STLForwardCompat.h"

namespace cir {

enum class FnInfoOpts {
  None = 0,
  IsInstanceMethod = 1 << 0,
  IsChainCall = 1 << 1,
  IsDelegateCall = 1 << 2,
};

inline FnInfoOpts operator|(FnInfoOpts a, FnInfoOpts b) {
  return static_cast<FnInfoOpts>(llvm::to_underlying(a) |
                                 llvm::to_underlying(b));
}

inline FnInfoOpts operator&(FnInfoOpts a, FnInfoOpts b) {
  return static_cast<FnInfoOpts>(llvm::to_underlying(a) &
                                 llvm::to_underlying(b));
}

inline FnInfoOpts operator|=(FnInfoOpts a, FnInfoOpts b) {
  a = a | b;
  return a;
}

inline FnInfoOpts operator&=(FnInfoOpts a, FnInfoOpts b) {
  a = a & b;
  return a;
}

} // namespace cir

#endif // CIR_FNINFOOPTS_H
