//===------ CIRGenCUDARuntime.h - Interface to CUDA Runtimes -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA CIR generation. Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
#define LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H

#include "clang/Basic/Sanitizers.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

namespace clang {
class CUDAKernelCallExpr;
}

namespace clang::CIRGen {

class CIRGenFunction;
class CIRGenModule;
class FunctionArgList;
class RValue;
class ReturnValueSlot;

class CIRGenCUDARuntime {
protected:
  CIRGenModule &cgm;

public:
  // Global variable properties that must be passed to CUDA runtime.
  class DeviceVarFlags {
  public:
    enum DeviceVarKind {
      Variable, // Variable
      Surface,  // Builtin surface
      Texture,  // Builtin texture
    };

  private:
    LLVM_PREFERRED_TYPE(DeviceVarKind)
    unsigned Kind : 2;
    LLVM_PREFERRED_TYPE(bool)
    unsigned Extern : 1;
    LLVM_PREFERRED_TYPE(bool)
    unsigned Constant : 1; // Constant variable.
    LLVM_PREFERRED_TYPE(bool)
    unsigned Managed : 1; // Managed variable.
    LLVM_PREFERRED_TYPE(bool)
    unsigned Normalized : 1; // Normalized texture.
    int SurfTexType;         // Type of surface/texutre.

  public:
    DeviceVarFlags(DeviceVarKind K, bool E, bool C, bool M, bool N, int T)
        : Kind(K), Extern(E), Constant(C), Managed(M), Normalized(N),
          SurfTexType(T) {}

    DeviceVarKind getKind() const { return static_cast<DeviceVarKind>(Kind); }
    bool isExtern() const { return Extern; }
    bool isConstant() const { return Constant; }
    bool isManaged() const { return Managed; }
    bool isNormalized() const { return Normalized; }
    int getSurfTexType() const { return SurfTexType; }
  };

  CIRGenCUDARuntime(CIRGenModule &cgm) : cgm(cgm) {}
  virtual ~CIRGenCUDARuntime();

  virtual void emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                              FunctionArgList &args) = 0;

  virtual RValue emitCUDAKernelCallExpr(CIRGenFunction &cgf,
                                        const CUDAKernelCallExpr *expr,
                                        ReturnValueSlot retValue);
  virtual mlir::Operation *getKernelHandle(cir::FuncOp fn, GlobalDecl GD) = 0;
  /// Get kernel stub by kernel handle.
  virtual mlir::Operation *getKernelStub(mlir::Operation *handle) = 0;

  virtual void internalizeDeviceSideVar(const VarDecl *d,
                                        cir::GlobalLinkageKind &linkage) = 0;

  /// Check whether a variable is a device variable and register it if true.
  virtual void handleVarRegistration(const VarDecl *vd, cir::GlobalOp var) = 0;

  /// Returns function or variable name on device side even if the current
  /// compilation is for host.
  virtual std::string getDeviceSideName(const NamedDecl *nd) = 0;
};

CIRGenCUDARuntime *CreateNVCUDARuntime(CIRGenModule &cgm);

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
