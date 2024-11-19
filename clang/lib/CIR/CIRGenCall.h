//===----- CIRGenCall.h - Encapsulate calling convention details ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CIRGENCALL_H
#define LLVM_CLANG_LIB_CODEGEN_CIRGENCALL_H

#include "CIRGenValue.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Type.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/BuiltinOps.h"

namespace cir {
class CIRGenFunction;

/// Abstract information about a function or function prototype.
class CIRGenCalleeInfo {
  const clang::FunctionProtoType *CalleeProtoTy;
  clang::GlobalDecl CalleeDecl;

public:
  explicit CIRGenCalleeInfo() : CalleeProtoTy(nullptr), CalleeDecl() {}
  CIRGenCalleeInfo(clang::GlobalDecl calleeDecl)
      : CalleeProtoTy(nullptr), CalleeDecl(calleeDecl) {}

  const clang::FunctionProtoType *getCalleeFunctionProtoType() const {
    return CalleeProtoTy;
  }
  const clang::GlobalDecl getCalleeDecl() const { return CalleeDecl; }
};

/// All available information about a concrete callee.
class CIRGenCallee {
  enum class SpecialKind : uintptr_t {
    Invalid,
    Builtin,
    PsuedoDestructor,
    Virtual,

    Last = Virtual
  };

  struct BuiltinInfoStorage {
    const clang::FunctionDecl *Decl;
    unsigned ID;
  };
  struct PseudoDestructorInfoStorage {
    const clang::CXXPseudoDestructorExpr *Expr;
  };
  struct VirtualInfoStorage {
    const clang::CallExpr *CE;
    clang::GlobalDecl MD;
    Address Addr;
    mlir::FunctionType FTy;
  };

  SpecialKind KindOrFunctionPointer;

  union {
    CIRGenCalleeInfo AbstractInfo;
    BuiltinInfoStorage BuiltinInfo;
    PseudoDestructorInfoStorage PseudoDestructorInfo;
    VirtualInfoStorage VirtualInfo;
  };

  explicit CIRGenCallee(SpecialKind kind) : KindOrFunctionPointer(kind) {}

public:
  CIRGenCallee() : KindOrFunctionPointer(SpecialKind::Invalid) {}

  // Construct a callee. Call this constructor directly when this isn't a direct
  // call.
  CIRGenCallee(const CIRGenCalleeInfo &abstractInfo, mlir::FuncOp functionPtr)
      : KindOrFunctionPointer(SpecialKind(
            reinterpret_cast<uintptr_t>(functionPtr.getAsOpaquePointer()))) {
    AbstractInfo = abstractInfo;
    assert(functionPtr && "configuring callee without function pointer");
    // TODO: codegen asserts functionPtr is a pointer
    // TODO: codegen asserts functionPtr is either an opaque pointer type or a
    // pointer to a function
  }

  static CIRGenCallee
  forDirect(mlir::FuncOp functionPtr,
            const CIRGenCalleeInfo &abstractInfo = CIRGenCalleeInfo()) {
    return CIRGenCallee(abstractInfo, functionPtr);
  }

  bool isBuiltin() const {
    return KindOrFunctionPointer == SpecialKind::Builtin;
  }

  bool isPsuedoDestructor() const {
    return KindOrFunctionPointer == SpecialKind::PsuedoDestructor;
  }

  bool isOrdinary() const {
    return uintptr_t(KindOrFunctionPointer) > uintptr_t(SpecialKind::Last);
  }

  /// If this is a delayed callee computation of some sort, prepare a concrete
  /// callee
  CIRGenCallee prepareConcreteCallee(CIRGenFunction &CGF) const;

  mlir::FuncOp getFunctionPointer() const {
    assert(isOrdinary());
    return mlir::FuncOp::getFromOpaquePointer(
        reinterpret_cast<void *>(KindOrFunctionPointer));
  }

  CIRGenCalleeInfo getAbstractInfo() const {
    assert(!isVirtual() && "Virtual NYI");
    assert(isOrdinary());
    return AbstractInfo;
  }

  bool isVirtual() const {
    return KindOrFunctionPointer == SpecialKind::Virtual;
  }
};

struct CallArg {
private:
  union {
    RValue RV;
    LValue LV; /// This argument is semantically a load from this l-value
  };
  bool HasLV;

  /// A data-flow flag to make sure getRValue and/or copyInto are not
  /// called twice for duplicated IR emission.
  mutable bool IsUsed;

public:
  clang::QualType Ty;
  CallArg(RValue rv, clang::QualType ty)
      : RV(rv), HasLV(false), IsUsed(false), Ty(ty) {
    (void)HasLV;
    (void)IsUsed;
  }
};

class CallArgList : public llvm::SmallVector<CallArg, 8> {
public:
  CallArgList() {}

  struct Writeback {
    LValue Source;
  };

  void add(RValue rvalue, clang::QualType type) {
    push_back(CallArg(rvalue, type));
  }
};

/// FunctionArgList - Type for representing both the decl and type of parameters
/// to a function. The decl must be either a ParmVarDecl or ImplicitParamDecl.
class FunctionArgList : public llvm::SmallVector<const clang::VarDecl *, 16> {};

/// ReturnValueSlot - Contains the address where the return value of a function
/// can be stored, and whether the address is volatile or not.
class ReturnValueSlot {
  Address Addr = Address::invalid();

  // Return value slot flags
  // unsigned IsVolatile : 1;
  // unsigned IsUnused : 1;
  // unsigned IsExternallyDestructed : 1;

public:
  // :
  ReturnValueSlot()
  //   IsVolatile(false),
  //   IsUnused(false),
  //   IsExternallyDestructed(false)
  {}
};

} // namespace cir

#endif
