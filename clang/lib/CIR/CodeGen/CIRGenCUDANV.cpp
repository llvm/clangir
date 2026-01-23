//===- CIRGenCUDANV.cpp - Interface to NVIDIA CUDA Runtime -----===//
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

#include "CIRGenCUDARuntime.h"
#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "mlir/IR/Operation.h"
#include "clang/Basic/Cuda.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::CIRGen;

static std::unique_ptr<MangleContext> initDeviceMC(CIRGenModule &cgm) {
  // If the host and device have different C++ ABIs, mark it as the device
  // mangle context so that the mangling needs to retrieve the additional
  // device lambda mangling number instead of the regular host one.
  if (cgm.getASTContext().getAuxTargetInfo() &&
      cgm.getASTContext().getTargetInfo().getCXXABI().isMicrosoft() &&
      cgm.getASTContext().getAuxTargetInfo()->getCXXABI().isItaniumFamily()) {
    return std::unique_ptr<MangleContext>(
        cgm.getASTContext().createDeviceMangleContext(
            *cgm.getASTContext().getAuxTargetInfo()));
  }

  return std::unique_ptr<MangleContext>(cgm.getASTContext().createMangleContext(
      cgm.getASTContext().getAuxTargetInfo()));
}

namespace {

class CIRGenNVCUDARuntime : public CIRGenCUDARuntime {
protected:
  StringRef Prefix;

  // Map a device stub function to a symbol for identifying kernel in host
  // code. For CUDA, the symbol for identifying the kernel is the same as the
  // device stub function. For HIP, they are different.
  llvm::DenseMap<StringRef, mlir::Operation *> KernelHandles;

  // Map a kernel handle to the kernel stub.
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> KernelStubs;
  // Mangle context for device.
  std::unique_ptr<MangleContext> deviceMC;

private:
  void emitDeviceStubBodyLegacy(CIRGenFunction &cgf, cir::FuncOp fn,
                                FunctionArgList &args);
  void emitDeviceStubBodyNew(CIRGenFunction &cgf, cir::FuncOp fn,
                             FunctionArgList &args);
  std::string addPrefixToName(StringRef FuncName) const;
  std::string addUnderscoredPrefixToName(StringRef FuncName) const;

public:
  CIRGenNVCUDARuntime(CIRGenModule &cgm);
  ~CIRGenNVCUDARuntime();

  void emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                      FunctionArgList &args) override;
  void handleVarRegistration(const VarDecl *vd, cir::GlobalOp var) override;
  void finalizeModule() override;

  mlir::Operation *getKernelHandle(cir::FuncOp fn, GlobalDecl GD) override;

  mlir::Operation *getKernelStub(mlir::Operation *handle) override {
    auto loc = KernelStubs.find(handle);
    assert(loc != KernelStubs.end());
    return loc->second;
  }

  void internalizeDeviceSideVar(const VarDecl *d,
                                cir::GlobalLinkageKind &linkage) override;
  /// Returns function or variable name on device side even if the current
  /// compilation is for host.
  std::string getDeviceSideName(const NamedDecl *nd) override;

  void registerDeviceVar(const VarDecl *vd, cir::GlobalOp &var, bool isExtern,
                         bool isConstant) {

    // Attach the device var attribute to the GlobalOp
    auto &builder = cgm.getBuilder();
    var->setAttr(cir::CUDAVarRegistrationInfoAttr::getMnemonic(),
                 cir::CUDAVarRegistrationInfoAttr::get(
                     builder.getContext(), cir::CUDADeviceVarKind::Variable,
                     isExtern, isConstant, vd->hasAttr<HIPManagedAttr>()));
  }
};

} // namespace

CIRGenCUDARuntime *clang::CIRGen::CreateNVCUDARuntime(CIRGenModule &cgm) {
  return new CIRGenNVCUDARuntime(cgm);
}

CIRGenNVCUDARuntime::~CIRGenNVCUDARuntime() {}

CIRGenNVCUDARuntime::CIRGenNVCUDARuntime(CIRGenModule &cgm)
    : CIRGenCUDARuntime(cgm), deviceMC(initDeviceMC(cgm)) {
  if (cgm.getLangOpts().OffloadViaLLVM)
    llvm_unreachable("NYI");
  else if (cgm.getLangOpts().HIP)
    Prefix = "hip";
  else
    Prefix = "cuda";
}

std::string CIRGenNVCUDARuntime::addPrefixToName(StringRef FuncName) const {
  return (Prefix + FuncName).str();
}
std::string
CIRGenNVCUDARuntime::addUnderscoredPrefixToName(StringRef FuncName) const {
  return ("__" + Prefix + FuncName).str();
}

void CIRGenNVCUDARuntime::emitDeviceStubBodyLegacy(CIRGenFunction &cgf,
                                                   cir::FuncOp fn,
                                                   FunctionArgList &args) {
  llvm_unreachable("NYI");
}

void CIRGenNVCUDARuntime::emitDeviceStubBodyNew(CIRGenFunction &cgf,
                                                cir::FuncOp fn,
                                                FunctionArgList &args) {

  // This requires arguments to be sent to kernels in a different way.
  if (cgm.getLangOpts().OffloadViaLLVM)
    llvm_unreachable("NYI");

  auto &builder = cgm.getBuilder();

  // For [cuda|hip]LaunchKernel, we must add another layer of indirection
  // to arguments. For example, for function `add(int a, float b)`,
  // we need to pass it as `void *args[2] = { &a, &b }`.

  auto loc = fn.getLoc();
  auto voidPtrArrayTy = cir::ArrayType::get(cgm.VoidPtrTy, args.size());
  mlir::Value kernelArgs = builder.createAlloca(
      loc, cir::PointerType::get(voidPtrArrayTy), voidPtrArrayTy, "kernel_args",
      CharUnits::fromQuantity(16));

  mlir::Value kernelArgsDecayed =
      builder.createCast(cir::CastKind::array_to_ptrdecay, kernelArgs,
                         cir::PointerType::get(cgm.VoidPtrTy));

  // Store arguments into kernelArgs
  for (auto [i, arg] : llvm::enumerate(args)) {
    mlir::Value index =
        builder.getConstInt(loc, llvm::APInt(/*numBits=*/32, i));
    mlir::Value storePos =
        builder.createPtrStride(loc, kernelArgsDecayed, index);
    builder.CIRBaseBuilderTy::createStore(
        loc, cgf.GetAddrOfLocalVar(arg).getPointer(), storePos);
  }

  // We retrieve dim3 type by looking into the second argument of
  // cudaLaunchKernel, as is done in OG.
  TranslationUnitDecl *tuDecl = cgm.getASTContext().getTranslationUnitDecl();
  DeclContext *dc = TranslationUnitDecl::castToDeclContext(tuDecl);

  // The default stream is usually stream 0 (the legacy default stream).
  // For per-thread default stream, we need a different LaunchKernel function.
  std::string kernelLaunchAPI = "LaunchKernel";
  if (cgm.getLangOpts().GPUDefaultStream ==
      LangOptions::GPUDefaultStreamKind::PerThread) {
    if (cgf.getLangOpts().HIP)
      kernelLaunchAPI = kernelLaunchAPI + "_spt";
    else if (cgf.getLangOpts().CUDA)
      kernelLaunchAPI = kernelLaunchAPI + "_ptsz";
  }

  std::string launchKernelName = addPrefixToName(kernelLaunchAPI);
  const IdentifierInfo &launchII =
      cgm.getASTContext().Idents.get(launchKernelName);
  FunctionDecl *launchFD = nullptr;
  for (auto *result : dc->lookup(&launchII)) {
    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(result))
      launchFD = fd;
  }

  if (launchFD == nullptr) {
    cgm.Error(cgf.CurFuncDecl->getLocation(),
              "Can't find declaration for " + launchKernelName);
    return;
  }

  // Use this function to retrieve arguments for cudaLaunchKernel:
  // int __[cuda|hip]PopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t
  //                                *sharedMem, cudaStream_t *stream)
  //
  // Here [cuda|hip]Stream_t, while also being the 6th argument of
  // [cuda|hip]LaunchKernel, is a pointer to some opaque struct.

  mlir::Type dim3Ty =
      cgf.getTypes().convertType(launchFD->getParamDecl(1)->getType());
  mlir::Type streamTy =
      cgf.getTypes().convertType(launchFD->getParamDecl(5)->getType());

  mlir::Value gridDim =
      builder.createAlloca(loc, cir::PointerType::get(dim3Ty), dim3Ty,
                           "grid_dim", CharUnits::fromQuantity(8));
  mlir::Value blockDim =
      builder.createAlloca(loc, cir::PointerType::get(dim3Ty), dim3Ty,
                           "block_dim", CharUnits::fromQuantity(8));
  mlir::Value sharedMem =
      builder.createAlloca(loc, cir::PointerType::get(cgm.SizeTy), cgm.SizeTy,
                           "shared_mem", cgm.getSizeAlign());
  mlir::Value stream =
      builder.createAlloca(loc, cir::PointerType::get(streamTy), streamTy,
                           "stream", cgm.getPointerAlign());

  cir::FuncOp popConfig = cgm.createRuntimeFunction(
      cir::FuncType::get({gridDim.getType(), blockDim.getType(),
                          sharedMem.getType(), stream.getType()},
                         cgm.SInt32Ty),
      addUnderscoredPrefixToName("PopCallConfiguration"));
  cgf.emitRuntimeCall(loc, popConfig, {gridDim, blockDim, sharedMem, stream});

  // Now emit the call to cudaLaunchKernel
  // [cuda|hip]Error_t [cuda|hip]LaunchKernel(const void *func, dim3 gridDim,
  // dim3 blockDim,
  //                              void **args, size_t sharedMem,
  //                              [cuda|hip]Stream_t stream);

  // We now either pick the function or the stub global for cuda, hip
  // resepectively.
  auto kernel = [&]() {
    if (auto globalOp = llvm::dyn_cast_or_null<cir::GlobalOp>(
            KernelHandles[fn.getSymName()])) {
      auto kernelTy = cir::PointerType::get(globalOp.getSymType());
      mlir::Value kernel = cir::GetGlobalOp::create(builder, loc, kernelTy,
                                                    globalOp.getSymName());
      return kernel;
    }
    if (auto funcOp = llvm::dyn_cast_or_null<cir::FuncOp>(
            KernelHandles[fn.getSymName()])) {
      auto kernelTy = cir::PointerType::get(funcOp.getFunctionType());
      mlir::Value kernel =
          cir::GetGlobalOp::create(builder, loc, kernelTy, funcOp.getSymName());
      mlir::Value func = builder.createBitcast(kernel, cgm.VoidPtrTy);
      return func;
    }
    assert(false && "Expected stub handle to be cir::GlobalOp or funcOp");
  }();
  // mlir::Value func = builder.createBitcast(kernel, cgm.VoidPtrTy);
  CallArgList launchArgs;

  launchArgs.add(RValue::get(kernel), launchFD->getParamDecl(0)->getType());
  launchArgs.add(
      RValue::getAggregate(Address(gridDim, CharUnits::fromQuantity(8))),
      launchFD->getParamDecl(1)->getType());
  launchArgs.add(
      RValue::getAggregate(Address(blockDim, CharUnits::fromQuantity(8))),
      launchFD->getParamDecl(2)->getType());
  launchArgs.add(RValue::get(kernelArgsDecayed),
                 launchFD->getParamDecl(3)->getType());
  launchArgs.add(
      RValue::get(builder.CIRBaseBuilderTy::createLoad(loc, sharedMem)),
      launchFD->getParamDecl(4)->getType());
  launchArgs.add(RValue::get(builder.CIRBaseBuilderTy::createLoad(loc, stream)),
                 launchFD->getParamDecl(5)->getType());

  mlir::Type launchTy = cgm.getTypes().convertType(launchFD->getType());
  mlir::Operation *launchFn = cgm.createRuntimeFunction(
      cast<cir::FuncType>(launchTy), launchKernelName);
  const auto &callInfo = cgm.getTypes().arrangeFunctionDeclaration(launchFD);
  cgf.emitCall(callInfo, CIRGenCallee::forDirect(launchFn), ReturnValueSlot(),
               launchArgs);
}

void CIRGenNVCUDARuntime::emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                                         FunctionArgList &args) {

  if (auto globalOp =
          llvm::dyn_cast<cir::GlobalOp>(KernelHandles[fn.getSymName()])) {
    auto &builder = cgm.getBuilder();
    auto fnPtrTy = globalOp.getSymType();
    auto sym = mlir::FlatSymbolRefAttr::get(fn.getSymNameAttr());
    auto gv = cir::GlobalViewAttr::get(fnPtrTy, sym);

    globalOp->setAttr("initial_value", gv);
    globalOp->removeAttr("sym_visibility");
    globalOp->setAttr("alignment", builder.getI64IntegerAttr(
                                       cgm.getPointerAlign().getQuantity()));
  }

  // CUDA 9.0 changed the way to launch kernels.
  if (CudaFeatureEnabled(cgm.getTarget().getSDKVersion(),
                         CudaFeature::CUDA_USES_NEW_LAUNCH) ||
      (cgm.getLangOpts().HIP && cgm.getLangOpts().HIPUseNewLaunchAPI) ||
      cgm.getLangOpts().OffloadViaLLVM)
    emitDeviceStubBodyNew(cgf, fn, args);
  else
    emitDeviceStubBodyLegacy(cgf, fn, args);
}

mlir::Operation *CIRGenNVCUDARuntime::getKernelHandle(cir::FuncOp fn,
                                                      GlobalDecl GD) {

  // Check if we already have a kernel handle for this function
  auto Loc = KernelHandles.find(fn.getSymName());
  if (Loc != KernelHandles.end()) {
    auto OldHandle = Loc->second;
    // Here we know that the fn did not change. Return it
    if (KernelStubs[OldHandle] == fn)
      return OldHandle;

    // We've found the function name, but F itself has changed, so we need to
    // update the references.
    if (cgm.getLangOpts().HIP) {
      // For HIP compilation the handle itself does not change, so we only need
      // to update the Stub value.
      KernelStubs[OldHandle] = fn;
      return OldHandle;
    }
    // For non-HIP compilation, erase the old Stub and fall-through to creating
    // new entries.
    KernelStubs.erase(OldHandle);
  }

  // If not targeting HIP, store the function itself
  if (!cgm.getLangOpts().HIP) {
    KernelHandles[fn.getSymName()] = fn;
    KernelStubs[fn] = fn;
    return fn;
  }

  // Create a new CIR global variable to represent the kernel handle
  auto &builder = cgm.getBuilder();
  auto globalName = cgm.getMangledName(
      GD.getWithKernelReferenceKind(KernelReferenceKind::Kernel));
  auto globalOp = cgm.getOrInsertGlobal(
      fn->getLoc(), globalName, fn.getFunctionType(), [&] {
        return CIRGenModule::createGlobalOp(
            cgm, fn->getLoc(), globalName,
            builder.getPointerTo(fn.getFunctionType()), true,
            /*addrSpace=*/{},
            /*insertPoint=*/nullptr);
      });

  globalOp->setAttr("alignment", builder.getI64IntegerAttr(
                                     cgm.getPointerAlign().getQuantity()));

  // Store references
  KernelHandles[fn.getSymName()] = globalOp;
  KernelStubs[globalOp] = fn;

  return globalOp;
}

std::string CIRGenNVCUDARuntime::getDeviceSideName(const NamedDecl *nd) {
  GlobalDecl gd;
  // nd could be either a kernel or a variable.
  if (auto *fd = dyn_cast<FunctionDecl>(nd))
    gd = GlobalDecl(fd, KernelReferenceKind::Kernel);
  else
    gd = GlobalDecl(nd);
  std::string deviceSideName;
  MangleContext *mc;
  if (cgm.getLangOpts().CUDAIsDevice)
    mc = &cgm.getCXXABI().getMangleContext();
  else
    mc = deviceMC.get();
  if (mc->shouldMangleDeclName(nd)) {
    SmallString<256> buffer;
    llvm::raw_svector_ostream out(buffer);
    mc->mangleName(gd, out);
    deviceSideName = std::string(out.str());
  } else
    deviceSideName = std::string(nd->getIdentifier()->getName());

  // Make unique name for device side static file-scope variable for HIP.
  if (cgm.getASTContext().shouldExternalize(nd) &&
      cgm.getLangOpts().GPURelocatableDeviceCode) {
    SmallString<256> buffer;
    llvm::raw_svector_ostream out(buffer);
    out << deviceSideName;
    cgm.printPostfixForExternalizedDecl(out, nd);
    deviceSideName = std::string(out.str());
  }
  return deviceSideName;
}

void CIRGenNVCUDARuntime::internalizeDeviceSideVar(
    const VarDecl *d, cir::GlobalLinkageKind &linkage) {
  if (cgm.getLangOpts().GPURelocatableDeviceCode)
    llvm_unreachable("NYI");

  // __shared__ variables are odd. Shadows do get created, but
  // they are not registered with the CUDA runtime, so they
  // can't really be used to access their device-side
  // counterparts. It's not clear yet whether it's nvcc's bug or
  // a feature, but we've got to do the same for compatibility.
  if (d->hasAttr<CUDADeviceAttr>() || d->hasAttr<CUDAConstantAttr>() ||
      d->hasAttr<CUDASharedAttr>()) {
    linkage = cir::GlobalLinkageKind::InternalLinkage;
  }

  if (d->getType()->isCUDADeviceBuiltinSurfaceType() ||
      d->getType()->isCUDADeviceBuiltinTextureType())
    llvm_unreachable("NYI");
}

void CIRGenNVCUDARuntime::handleVarRegistration(const VarDecl *declaration,
                                                cir::GlobalOp globalVariable) {
  if (declaration->hasAttr<CUDADeviceAttr>() ||
      declaration->hasAttr<CUDAConstantAttr>()) {
    // Shadow variables and their properties must be registered with CUDA
    // runtime. Skip Extern global variables, which will be registered in
    // the TU where they are defined.
    //
    // Don't register a C++17 inline variable. The local symbol can be
    // discarded and referencing a discarded local symbol from outside the
    // comdat (__cuda_register_globals) is disallowed by the ELF spec.
    //
    // HIP managed variables need to be always recorded in device and host
    // compilations for transformation.
    //
    // HIP managed variables and variables in CUDADeviceVarODRUsedByHost are
    // added to llvm.compiler-used, therefore they are safe to be registered.
    if ((!declaration->hasExternalStorage() && !declaration->isInline()) ||
        cgm.getASTContext().CUDADeviceVarODRUsedByHost.contains(declaration) ||
        declaration->hasAttr<HIPManagedAttr>()) {
      registerDeviceVar(declaration, globalVariable,
                        !declaration->hasDefinition(),
                        declaration->hasAttr<CUDAConstantAttr>());
    }
  } else if (declaration->getType()->isCUDADeviceBuiltinSurfaceType() ||
             declaration->getType()->isCUDADeviceBuiltinTextureType()) {
    // Builtin surfaces and textures and their template arguments are
    // also registered with CUDA runtime.
    llvm_unreachable("Surface and Texture registration NYI");
  }
}

void CIRGenNVCUDARuntime::finalizeModule() {
  if (!cgm.getLangOpts().CUDAIsDevice)
    return;

  // Mark ODR-used device variables as compiler used to prevent them from being
  // eliminated by optimization. This is necessary for device variables
  // ODR-used by host functions. Sema correctly marks them as ODR-used no
  // matter whether they are ODR-used by device or host functions.
  //
  // We do not need to do this if the variable has used attribute since it
  // has already been added.
  //
  // Static device variables have been externalized at this point, therefore
  // variables with private or internal linkage need not be added.
  for (auto globalOp : cgm.getModule().getOps<cir::GlobalOp>()) {
    auto regAttr = globalOp->getAttrOfType<cir::CUDAVarRegistrationInfoAttr>(
        cir::CUDAVarRegistrationInfoAttr::getMnemonic());
    if (!regAttr)
      continue;

    auto kind = regAttr.getKind();
    if (!globalOp.isDeclaration() &&
        !cir::isLocalLinkage(globalOp.getLinkage()) &&
        (kind == cir::CUDADeviceVarKind::Variable ||
         kind == cir::CUDADeviceVarKind::Surface ||
         kind == cir::CUDADeviceVarKind::Texture)) {
      cgm.addCompilerUsedGlobal(globalOp);
    }
  }
}
