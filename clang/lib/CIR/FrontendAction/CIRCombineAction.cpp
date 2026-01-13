#include "clang/CIR/FrontendAction/CIRCombineAction.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CodeGen/BackendUtil.h"

using namespace cir;
using namespace clang;

static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
loadModule(llvm::MemoryBufferRef MbRef, mlir::MLIRContext &mlirContext) {
  auto Module =
      mlir::parseSourceString<mlir::ModuleOp>(MbRef.getBuffer(), &mlirContext);
  if (!Module)
    return mlir::failure();
  return Module;
}

CIRCombineAction::CIRCombineAction() : mlirContext(new mlir::MLIRContext) {}

void CIRCombineAction::ExecuteAction() {
  auto &Ci = getCompilerInstance();
  auto &Diags = Ci.getDiagnostics();
  const clang::FrontendOptions &Fo = Ci.getFrontendOpts();

  // Expect ParseFrontendArgs already validated these, but keep it defensive.
  if (Fo.CIRHostInput.empty() || Fo.CIRDeviceInput.empty()) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << "missing -cir-host-input/-cir-device-input";
    return;
  }

  if (!Fo.EmitSplit && Fo.OutputFile.empty()) {
    Diags.Report(clang::diag::err_drv_missing_arg_mtp) << "-o";
    return;
  } else if (Fo.EmitSplit) {
    if (Fo.CIRHostOutput.empty())
      Diags.Report(clang::diag::err_drv_missing_arg_mtp) << "-cir-host-output";
    if (Fo.CIRDeviceOutput.empty())
      Diags.Report(clang::diag::err_drv_missing_arg_mtp)
          << "-cir-device-output";
  }

  mlirContext->getOrLoadDialect<mlir::BuiltinDialect>();
  mlirContext->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlirContext->getOrLoadDialect<mlir::DLTIDialect>();
  mlirContext->getOrLoadDialect<cir::CIRDialect>();
  mlirContext->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlirContext->getOrLoadDialect<mlir::arith::ArithDialect>();
  mlirContext->getOrLoadDialect<mlir::omp::OpenMPDialect>();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> HostInputOrErr =
      Ci.getFileManager().getBufferForFile(Fo.CIRHostInput);
  if (!HostInputOrErr) {
    std::error_code Ec = HostInputOrErr.getError();
    Diags.Report(clang::diag::err_fe_error_reading) << Fo.CIRHostInput;
    Diags.Report(clang::diag::note_drv_command_failed_diag_msg) << Ec.message();
    return;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> DeviceInputOrErr =
      Ci.getFileManager().getBufferForFile(Fo.CIRDeviceInput);
  if (!DeviceInputOrErr) {
    std::error_code Ec = DeviceInputOrErr.getError();
    Diags.Report(clang::diag::err_fe_error_reading) << Fo.CIRDeviceInput;
    Diags.Report(clang::diag::note_drv_command_failed_diag_msg) << Ec.message();
    return;
  }

  std::unique_ptr<llvm::MemoryBuffer> HostInput = std::move(*HostInputOrErr);
  std::unique_ptr<llvm::MemoryBuffer> DeviceInput =
      std::move(*DeviceInputOrErr);

  auto HostCirModuleOr = loadModule(*HostInput, *mlirContext);
  if (mlir::failed(HostCirModuleOr)) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << "failed to parse CIR module" << Fo.CIRHostInput;
    return;
  }
  mlir::OwningOpRef<mlir::ModuleOp> HostCirModule = std::move(*HostCirModuleOr);

  auto DeviceCirModuleOr = loadModule(*DeviceInput, *mlirContext);
  if (mlir::failed(DeviceCirModuleOr)) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << "failed to parse CIR module" << Fo.CIRDeviceInput;
    return;
  }
  mlir::OwningOpRef<mlir::ModuleOp> DeviceCirModule =
      std::move(*DeviceCirModuleOr);

  // Rename nested modules so the container verifier can find them.
  HostCirModule->setSymNameAttr(mlir::StringAttr::get(mlirContext, "host"));
  DeviceCirModule->setSymNameAttr(mlir::StringAttr::get(mlirContext, "device"));

  // Create a new top-level module which will hold the container.
  auto Loc = mlir::UnknownLoc::get(mlirContext);
  mlir::ModuleOp Combined = mlir::ModuleOp::create(Loc);

  mlir::OpBuilder Builder = mlir::OpBuilder::atBlockBegin(Combined.getBody());
  // Insert the container op into the top-level module body.
  Builder.setInsertionPointToStart(Combined.getBody());
  auto Container = cir::OffloadContainerOp::create(Builder, Loc);

  // Ensure the container region has a block.
  mlir::Region &Body = Container.getBody();
  if (Body.empty())
    Body.push_back(new mlir::Block());

  mlir::Block &Blk = Body.front();
  mlir::OpBuilder IB(&Blk, Blk.end());

  // Clone the parsed host/device modules into the container body.
  // (Cloning is simplest/robust for PR2.)
  IB.insert(HostCirModule->getOperation()->clone());
  IB.insert(DeviceCirModule->getOperation()->clone());
  auto EmitCIR = [&](mlir::ModuleOp &mOp, StringRef Output) {
    mlir::OpPrintingFlags Flags;
    Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/true);

    std::error_code EC;
    llvm::raw_fd_ostream OS(Output, EC, llvm::sys::fs::OF_Text);

    if (EC) {
      Diags.Report(clang::diag::err_fe_error_opening) << Output << EC.message();
      return;
    }

    mOp.print(OS, Flags);
  };

  if (!Fo.EmitSplit) {
    EmitCIR(Combined, Fo.OutputFile);
    return;
  }
  auto devModOr = Container.getDeviceModule();
  if (!devModOr)
    Container.emitError("missing device module in offload container");
  EmitCIR(*devModOr, Fo.CIRDeviceOutput);
  auto hostModOr = Container.getHostModule();
  if (!hostModOr)
    Container.emitError("missing host module in offload container");
  EmitCIR(*hostModOr, Fo.CIRHostOutput);
}
