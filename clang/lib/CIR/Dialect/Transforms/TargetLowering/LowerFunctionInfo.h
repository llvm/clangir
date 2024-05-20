//==-- LowerFunctionInfo.h - Represents of function argument/return types --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/inlcude/CodeGen/LowerFunctionInfo.h. The
// queries are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H

#include "MissingFeature.h"
#include "mlir/IR/Types.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/TrailingObjects.h"
#include <cstdint>

namespace mlir {
namespace cir {

/// Helper class to encapsulate information about how a
/// specific ABI-independent CIR type should be passed to or returned from a
/// function in an ABI-specific way.
class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Not yet supported.
    Direct,
    Ignore,
    Extend,
    Indirect,
    IndirectAliased,
    Expand,
    CoerceAndExpand,
    InAlloca,
    KindFirst = Direct,
    KindLast = InAlloca
  };

private:
  mlir::Type typeData;
  union {
    Type PaddingType;                 // canHavePaddingType()
    Type UnpaddedCoerceAndExpandType; // isCoerceAndExpand()
  };
  struct DirectAttrInfo {
    unsigned Offset;
    unsigned Align;
  };
  union {
    DirectAttrInfo DirectAttr; // isDirect() || isExtend()
  };
  Kind kind;
  bool CanBeFlattened : 1; // isDirect()
  bool InReg : 1;          // isDirect() || isExtend() || isIndirect()
  bool SignExt : 1;        // isExtend()

public:
  ABIArgInfo(Kind kind = Direct) : kind(kind), InReg(false), SignExt(false){};
  ~ABIArgInfo() = default;

  void setCanBeFlattened(bool Flatten) {
    assert(isDirect() && "Invalid kind!");
    CanBeFlattened = Flatten;
  }

  void setCoerceToType(Type T) {
    assert(canHaveCoerceToType() && "Invalid kind!");
    typeData = T;
  }

  void setDirectAlign(unsigned Align) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    DirectAttr.Align = Align;
  }

  Kind getKind() const { return kind; }
  bool isDirect() const { return kind == Direct; }
  bool isInAlloca() const { return kind == InAlloca; }
  bool isExtend() const { return kind == Extend; }
  bool isIndirect() const { return kind == Indirect; }
  bool isIndirectAliased() const { return kind == IndirectAliased; }
  bool isExpand() const { return kind == Expand; }
  bool isCoerceAndExpand() const { return kind == CoerceAndExpand; }

  bool isSignExt() const {
    assert(isExtend() && "Invalid kind!");
    return SignExt;
  }

  bool getInReg() const {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    return InReg;
  }
  void setInReg(bool IR) {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    InReg = IR;
  }

  bool canHaveCoerceToType() const {
    return isDirect() || isExtend() || isCoerceAndExpand();
  }
};

/// A class for recording the number of arguments that a function
/// signature requires.
class RequiredArgs {
  /// The number of required arguments, or ~0 if the signature does
  /// not permit optional arguments.
  unsigned NumRequired;

public:
  enum All_t { All };

  RequiredArgs(All_t _) : NumRequired(~0U) {}
  explicit RequiredArgs(unsigned n) : NumRequired(n) { assert(n != ~0U); }

  bool allowsOptionalArgs() const { return NumRequired != ~0U; }
};

// Implementation detail of LowerFunctionInfo, factored out so it can be
// named in the TrailingObjects base class of CGFunctionInfo.
struct LowerFunctionInfoArgInfo {
  mlir::Type type; // Original ABI-agnostic type.
  ABIArgInfo info; // ABI-specific information.
};

class LowerFunctionInfo final
    : private llvm::TrailingObjects<LowerFunctionInfo,
                                    LowerFunctionInfoArgInfo> {
  typedef LowerFunctionInfoArgInfo ArgInfo;

  /// The LLVM::CallingConv to use for this function (as specified by the
  /// user).
  unsigned CallingConvention : 8;

  /// The LLVM::CallingConv to actually use for this function, which may
  /// depend on the ABI.
  unsigned EffectiveCallingConvention : 8;

  /// Whether this is an instance method.
  unsigned InstanceMethod : 1;

  /// Whether this is a chain call.
  unsigned ChainCall : 1;

  /// Whether this function is called by forwarding arguments.
  /// This doesn't support inalloca or varargs.
  unsigned DelegateCall : 1;

  RequiredArgs Required;

  /// The struct representing all arguments passed in memory.  Only used when
  /// passing non-trivial types with inalloca.  Not part of the profile.
  StructType ArgStruct;

  unsigned NumArgs;

  const ArgInfo *getArgsBuffer() const { return getTrailingObjects<ArgInfo>(); }
  ArgInfo *getArgsBuffer() { return getTrailingObjects<ArgInfo>(); }

  LowerFunctionInfo() : Required(RequiredArgs::All) {}

public:
  static LowerFunctionInfo *create(unsigned llvmCC, bool instanceMethod,
                                   bool chainCall, bool delegateCall,
                                   Type resultType,
                                   ArrayRef<mlir::Type> argTypes,
                                   RequiredArgs required) {
    // TODO(cir): Add assertions?
    assert(MissingFeature::extParamInfo());
    void *buffer = operator new(totalSizeToAlloc<ArgInfo>(argTypes.size() + 1));

    LowerFunctionInfo *FI = new (buffer) LowerFunctionInfo();
    FI->CallingConvention = llvmCC;
    FI->EffectiveCallingConvention = llvmCC;
    FI->InstanceMethod = instanceMethod;
    FI->ChainCall = chainCall;
    FI->DelegateCall = delegateCall;
    FI->Required = required;
    FI->ArgStruct = nullptr;
    FI->NumArgs = argTypes.size();
    FI->getArgsBuffer()[0].type = resultType;
    for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
      FI->getArgsBuffer()[i + 1].type = argTypes[i];

    return FI;
  };

  // Friending class TrailingObjects is apparently not good enough for MSVC,
  // so these have to be public.
  friend class TrailingObjects;
  size_t numTrailingObjects(OverloadToken<ArgInfo>) const {
    return NumArgs + 1;
  }

  unsigned arg_size() const { return NumArgs; }

  bool isVariadic() const {
    assert(MissingFeature::variadicFunctions());
    return false;
  }
  unsigned getNumRequiredArgs() const {
    if (isVariadic())
      llvm_unreachable("NYI");
    return arg_size();
  }
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERFUNCTIONINFO_H
