//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/FrontendAction/CIRGenAction.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/CIRToCIRPasses.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/Passes.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <memory>

using namespace cir;
using namespace clang;

static std::string sanitizePassOptions(llvm::StringRef O) {
  if (O.empty())
    return "";
  std::string Opts{O};
  // MLIR pass options are space separated, but we use ';' in clang since
  // space aren't well supported, switch it back.
  for (unsigned I = 0, E = Opts.size(); I < E; ++I)
    if (Opts[I] == ';')
      Opts[I] = ' ';
  // If arguments are surrounded with '"', trim them off
  return llvm::StringRef(Opts).trim('"').str();
}

namespace cir {

static BackendAction
getBackendActionFromOutputType(CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitAssembly:
    return BackendAction::Backend_EmitAssembly;
  case CIRGenAction::OutputType::EmitBC:
    return BackendAction::Backend_EmitBC;
  case CIRGenAction::OutputType::EmitLLVM:
    return BackendAction::Backend_EmitLL;
  case CIRGenAction::OutputType::EmitObj:
    return BackendAction::Backend_EmitObj;
  default:
    llvm_unreachable("Unsupported action");
  }
}

static std::unique_ptr<llvm::Module> lowerFromCIRToLLVMIR(
    const clang::FrontendOptions &FeOptions, mlir::ModuleOp MlirMod,
    std::unique_ptr<mlir::MLIRContext> MlirCtx, llvm::LLVMContext &LlvmCtx,
    bool DisableVerifier = false, bool DisableCcLowering = false,
    bool DisableDebugInfo = false) {
  if (FeOptions.ClangIRDirectLowering)
    return direct::lowerDirectlyFromCIRToLLVMIR(
        MlirMod, LlvmCtx, DisableVerifier, DisableCcLowering, DisableDebugInfo);
  else
    return lowerFromCIRToMLIRToLLVMIR(MlirMod, std::move(MlirCtx), LlvmCtx);
}

class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();

  CIRGenAction::OutputType Action;

  CompilerInstance &CompilerInstance;
  DiagnosticsEngine &DiagnosticsEngine;
  [[maybe_unused]] const HeaderSearchOptions &HeaderSearchOptions;
  CodeGenOptions &CodeGenOptions;
  [[maybe_unused]] const TargetOptions &TargetOptions;
  [[maybe_unused]] const LangOptions &LangOptions;
  const FrontendOptions &FeOptions;

  std::unique_ptr<raw_pwrite_stream> OutputStream;

  ASTContext *AstContext{nullptr};
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::unique_ptr<CIRGenerator> Gen;

public:
  CIRGenConsumer(CIRGenAction::OutputType Action,
                 class CompilerInstance &CompilerInstance,
                 class DiagnosticsEngine &DiagnosticsEngine,
                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                 const class HeaderSearchOptions &HeaderSearchOptions,
                 class CodeGenOptions &CodeGenOptions,
                 const class TargetOptions &TargetOptions,
                 const class LangOptions &LangOptions,
                 const FrontendOptions &FeOptions,
                 std::unique_ptr<raw_pwrite_stream> Os)
      : Action(Action), CompilerInstance(CompilerInstance),
        DiagnosticsEngine(DiagnosticsEngine),
        HeaderSearchOptions(HeaderSearchOptions),
        CodeGenOptions(CodeGenOptions), TargetOptions(TargetOptions),
        LangOptions(LangOptions), FeOptions(FeOptions),
        OutputStream(std::move(Os)), FS(VFS),
        Gen(std::make_unique<CIRGenerator>(DiagnosticsEngine, std::move(VFS),
                                           CodeGenOptions)) {}

  void Initialize(ASTContext &Ctx) override {
    assert(!AstContext && "initialized multiple times");

    AstContext = &Ctx;

    Gen->Initialize(Ctx);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                   AstContext->getSourceManager(),
                                   "LLVM IR generation of declaration");
    Gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    Gen->HandleCXXStaticMemberVarInstantiation(VD);
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    Gen->HandleInlineFunctionDefinition(D);
  }

  void HandleInterestingDecl(DeclGroupRef D) override {
    llvm_unreachable("NYI");
  }

  void HandleTranslationUnit(ASTContext &C) override {
    llvm::TimeTraceScope Scope("CIR Gen");

    // Note that this method is called after `HandleTopLevelDecl` has already
    // ran all over the top level decls. Here clang mostly wraps defered and
    // global codegen, followed by running CIR passes.
    Gen->HandleTranslationUnit(C);

    if (!FeOptions.ClangIRDisableCIRVerifier)
      if (!Gen->verifyModule()) {
        llvm::report_fatal_error(
            "CIR codegen: module verification error before running CIR passes");
        return;
      }

    auto MlirMod = Gen->getModule();
    auto MlirCtx = Gen->takeContext();

    auto SetupCirPipelineAndExecute = [&] {
      // Sanitize passes options. MLIR uses spaces between pass options
      // and since that's hard to fly in clang, we currently use ';'.
      std::string LifetimeOpts, IdiomRecognizerOpts, LibOptOpts;
      if (FeOptions.ClangIRLifetimeCheck)
        LifetimeOpts = sanitizePassOptions(FeOptions.ClangIRLifetimeCheckOpts);
      if (FeOptions.ClangIRIdiomRecognizer)
        IdiomRecognizerOpts =
            sanitizePassOptions(FeOptions.ClangIRIdiomRecognizerOpts);
      if (FeOptions.ClangIRLibOpt)
        LibOptOpts = sanitizePassOptions(FeOptions.ClangIRLibOptOpts);

      bool EnableCcLowering =
          FeOptions.ClangIRCallConvLowering &&
          !(Action == CIRGenAction::OutputType::EmitMLIR &&
            FeOptions.MLIRTargetDialect == frontend::MLIR_CIR);
      bool FlattenCir =
          Action == CIRGenAction::OutputType::EmitMLIR &&
          FeOptions.MLIRTargetDialect == clang::frontend::MLIR_CIR_FLAT;

      // Setup and run CIR pipeline.
      std::string PassOptParsingFailure;
      if (runCIRToCIRPasses(
              MlirMod, MlirCtx.get(), C, !FeOptions.ClangIRDisableCIRVerifier,
              FeOptions.ClangIRLifetimeCheck, LifetimeOpts,
              FeOptions.ClangIRIdiomRecognizer, IdiomRecognizerOpts,
              FeOptions.ClangIRLibOpt, LibOptOpts, PassOptParsingFailure,
              CodeGenOptions.OptimizationLevel > 0, FlattenCir,
              !FeOptions.ClangIRDirectLowering, EnableCcLowering,
              FeOptions.ClangIREnableMem2Reg)
              .failed()) {
        if (!PassOptParsingFailure.empty()) {
          auto D = DiagnosticsEngine.Report(diag::err_drv_cir_pass_opt_parsing);
          D << FeOptions.ClangIRLifetimeCheckOpts;
        } else
          llvm::report_fatal_error("CIR codegen: MLIR pass manager fails "
                                   "when running CIR passes!");
        return;
      }
    };

    if (!FeOptions.ClangIRDisablePasses) {
      // Handle source manager properly given that lifetime analysis
      // might emit warnings and remarks.
      auto &ClangSourceMgr = C.getSourceManager();
      FileID MainFileID = ClangSourceMgr.getMainFileID();

      std::unique_ptr<llvm::MemoryBuffer> FileBuf =
          llvm::MemoryBuffer::getMemBuffer(
              ClangSourceMgr.getBufferOrFake(MainFileID));

      llvm::SourceMgr MlirSourceMgr;
      MlirSourceMgr.AddNewSourceBuffer(std::move(FileBuf), llvm::SMLoc());

      if (FeOptions.ClangIRVerifyDiags) {
        mlir::SourceMgrDiagnosticVerifierHandler SourceMgrHandler(
            MlirSourceMgr, MlirCtx.get());
        MlirCtx->printOpOnDiagnostic(false);
        SetupCirPipelineAndExecute();

        // Verify the diagnostic handler to make sure that each of the
        // diagnostics matched.
        if (SourceMgrHandler.verify().failed()) {
          // FIXME: we fail ungracefully, there's probably a better way
          // to communicate non-zero return so tests can actually fail.
          llvm::sys::RunInterruptHandlers();
          exit(1);
        }
      } else {
        mlir::SourceMgrDiagnosticHandler SourceMgrHandler(MlirSourceMgr,
                                                          MlirCtx.get());
        SetupCirPipelineAndExecute();
      }
    }

    auto EmitMlir = [&](mlir::Operation *MlirMod, bool Verify) {
      assert(MlirMod &&
             "MLIR module does not exist, but lowering did not fail?");
      assert(OutputStream && "Why are we here without an output stream?");
      // FIXME: we cannot roundtrip prettyForm=true right now.
      mlir::OpPrintingFlags Flags;
      Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
      if (!Verify)
        Flags.assumeVerified();
      MlirMod->print(*OutputStream, Flags);
    };

    switch (Action) {
    case CIRGenAction::OutputType::EmitMLIR: {
      switch (FeOptions.MLIRTargetDialect) {
      case clang::frontend::MLIR_CORE:
        // case for direct lowering is already checked in compiler invocation
        // no need to check here
        EmitMlir(lowerFromCIRToMLIR(MlirMod, MlirCtx.get()), false);
        break;
      case clang::frontend::MLIR_LLVM: {
        mlir::ModuleOp LoweredMlirModule =
            FeOptions.ClangIRDirectLowering
                ? direct::lowerDirectlyFromCIRToLLVMDialect(MlirMod)
                : lowerFromCIRToMLIRToLLVMDialect(MlirMod, MlirCtx.get());
        EmitMlir(LoweredMlirModule, false);
        break;
      }
      case clang::frontend::MLIR_CIR:
      case clang::frontend::MLIR_CIR_FLAT:
        EmitMlir(MlirMod, FeOptions.ClangIRDisableCIRVerifier);
        break;
      }
      break;
    }
    case CIRGenAction::OutputType::EmitLLVM:
    case CIRGenAction::OutputType::EmitBC:
    case CIRGenAction::OutputType::EmitObj:
    case CIRGenAction::OutputType::EmitAssembly: {
      llvm::LLVMContext LlvmCtx;
      bool DisableDebugInfo =
          CodeGenOptions.getDebugInfo() == llvm::codegenoptions::NoDebugInfo;
      auto LlvmModule = lowerFromCIRToLLVMIR(
          FeOptions, MlirMod, std::move(MlirCtx), LlvmCtx,
          FeOptions.ClangIRDisableCIRVerifier,
          !FeOptions.ClangIRCallConvLowering, DisableDebugInfo);

      BackendAction BackendAction = getBackendActionFromOutputType(Action);

      emitBackendOutput(CompilerInstance, CodeGenOptions,
                        C.getTargetInfo().getDataLayoutString(),
                        LlvmModule.get(), BackendAction, FS,
                        std::move(OutputStream));
      break;
    }
    case CIRGenAction::OutputType::None:
      break;
    }
  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                   AstContext->getSourceManager(),
                                   "CIR generation of declaration");
    Gen->HandleTagDeclDefinition(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    Gen->HandleTagDeclRequiredDefinition(D);
  }

  void CompleteTentativeDefinition(VarDecl *D) override {
    Gen->CompleteTentativeDefinition(D);
  }

  void CompleteExternalDeclaration(DeclaratorDecl *D) override {
    llvm_unreachable("NYI");
  }

  void AssignInheritanceModel(CXXRecordDecl *RD) override {
    llvm_unreachable("NYI");
  }

  void HandleVTable(CXXRecordDecl *RD) override { Gen->HandleVTable(RD); }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType Act, mlir::MLIRContext *MlirContext)
    : mlirContext(MlirContext ? MlirContext : new mlir::MLIRContext),
      action(Act) {}

CIRGenAction::~CIRGenAction() { mlirModule.reset(); }

void CIRGenAction::EndSourceFileAction() {
  // If the consumer creation failed, do nothing.
  if (!getCompilerInstance().hasASTConsumer())
    return;

  // TODO: pass the module around
  // module = cgConsumer->takeModule();
}

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &Ci, StringRef InFile,
                CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitAssembly:
    return Ci.createDefaultOutputFile(false, InFile, "s");
  case CIRGenAction::OutputType::EmitMLIR:
    return Ci.createDefaultOutputFile(false, InFile, "mlir");
  case CIRGenAction::OutputType::EmitLLVM:
    return Ci.createDefaultOutputFile(false, InFile, "ll");
  case CIRGenAction::OutputType::EmitBC:
    return Ci.createDefaultOutputFile(true, InFile, "bc");
  case CIRGenAction::OutputType::EmitObj:
    return Ci.createDefaultOutputFile(true, InFile, "o");
  case CIRGenAction::OutputType::None:
    return nullptr;
  }

  llvm_unreachable("Invalid action!");
}

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &Ci, StringRef InputFile) {
  auto Out = Ci.takeOutputStream();
  if (!Out)
    Out = getOutputStream(Ci, InputFile, action);

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      action, Ci, Ci.getDiagnostics(), &Ci.getVirtualFileSystem(),
      Ci.getHeaderSearchOpts(), Ci.getCodeGenOpts(), Ci.getTargetOpts(),
      Ci.getLangOpts(), Ci.getFrontendOpts(), std::move(Out));
  cgConsumer = Result.get();

  // Enable generating macro debug info only when debug info is not disabled and
  // also macrod ebug info is enabled
  if (Ci.getCodeGenOpts().getDebugInfo() != llvm::codegenoptions::NoDebugInfo &&
      Ci.getCodeGenOpts().MacroDebugInfo) {
    llvm_unreachable("NYI");
  }

  return std::move(Result);
}

mlir::OwningOpRef<mlir::ModuleOp>
CIRGenAction::loadModule(llvm::MemoryBufferRef MbRef) {
  auto Module =
      mlir::parseSourceString<mlir::ModuleOp>(MbRef.getBuffer(), mlirContext);
  assert(Module && "Failed to parse ClangIR module");
  return Module;
}

void CIRGenAction::ExecuteAction() {
  if (getCurrentFileKind().getLanguage() != Language::CIR) {
    this->ASTFrontendAction::ExecuteAction();
    return;
  }

  // If this is a CIR file we have to treat it specially.
  // TODO: This could be done more logically. This is just modeled at the moment
  // mimicing CodeGenAction but this is clearly suboptimal.
  auto &Ci = getCompilerInstance();
  std::unique_ptr<raw_pwrite_stream> Outstream =
      getOutputStream(Ci, getCurrentFile(), action);
  if (action != OutputType::None && !Outstream)
    return;

  auto &SourceManager = Ci.getSourceManager();
  auto FileId = SourceManager.getMainFileID();
  auto MainFile = SourceManager.getBufferOrNone(FileId);

  if (!MainFile)
    return;

  mlirContext->getOrLoadDialect<cir::CIRDialect>();
  mlirContext->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();

  // TODO: unwrap this -- this exists because including the `OwningModuleRef` in
  // CIRGenAction's header would require linking the Frontend against MLIR.
  // Let's avoid that for now.
  auto MlirModule = loadModule(*MainFile);
  if (!MlirModule)
    return;

  // FIXME(cir): This compilation path does not account for some flags.
  llvm::LLVMContext LlvmCtx;
  bool DisableDebugInfo =
      Ci.getCodeGenOpts().getDebugInfo() == llvm::codegenoptions::NoDebugInfo;
  auto LlvmModule = lowerFromCIRToLLVMIR(
      Ci.getFrontendOpts(), MlirModule.release(),
      std::unique_ptr<mlir::MLIRContext>(mlirContext), LlvmCtx,
      /*disableVerifier=*/false, /*disableCCLowering=*/true, DisableDebugInfo);

  if (Outstream)
    LlvmModule->print(*Outstream, nullptr);
}

namespace cir {
void EmitAssemblyAction::anchor() {}
EmitAssemblyAction::EmitAssemblyAction(mlir::MLIRContext *MlirContext)
    : CIRGenAction(OutputType::EmitAssembly, MlirContext) {}

void EmitCIROnlyAction::anchor() {}
EmitCIROnlyAction::EmitCIROnlyAction(mlir::MLIRContext *MlirContext)
    : CIRGenAction(OutputType::None, MlirContext) {}

void EmitMLIRAction::anchor() {}
EmitMLIRAction::EmitMLIRAction(mlir::MLIRContext *MlirContext)
    : CIRGenAction(OutputType::EmitMLIR, MlirContext) {}

void EmitLLVMAction::anchor() {}
EmitLLVMAction::EmitLLVMAction(mlir::MLIRContext *MlirContext)
    : CIRGenAction(OutputType::EmitLLVM, MlirContext) {}

void EmitBCAction::anchor() {}
EmitBCAction::EmitBCAction(mlir::MLIRContext *MlirContext)
    : CIRGenAction(OutputType::EmitBC, MlirContext) {}

void EmitObjAction::anchor() {}
EmitObjAction::EmitObjAction(mlir::MLIRContext *MlirContext)
    : CIRGenAction(OutputType::EmitObj, MlirContext) {}
} // namespace cir

// Used for -fclangir-analysis-only: use CIR analysis but still use original
// LLVM codegen path
void AnalysisOnlyActionBase::anchor() {}
AnalysisOnlyActionBase::AnalysisOnlyActionBase(unsigned Act,
                                               llvm::LLVMContext *VmContext)
    : clang::CodeGenAction(Act, VmContext) {}

std::unique_ptr<ASTConsumer>
AnalysisOnlyActionBase::CreateASTConsumer(clang::CompilerInstance &Ci,
                                          llvm::StringRef InFile) {
  std::vector<std::unique_ptr<ASTConsumer>> Consumers;
  Consumers.push_back(clang::CodeGenAction::CreateASTConsumer(Ci, InFile));
  Consumers.push_back(std::make_unique<cir::CIRGenConsumer>(
      CIRGenAction::OutputType::None, Ci, Ci.getDiagnostics(),
      &Ci.getVirtualFileSystem(), Ci.getHeaderSearchOpts(), Ci.getCodeGenOpts(),
      Ci.getTargetOpts(), Ci.getLangOpts(), Ci.getFrontendOpts(), nullptr));
  return std::make_unique<MultiplexConsumer>(std::move(Consumers));
}

void AnalysisOnlyAndEmitAssemblyAction::anchor() {}
AnalysisOnlyAndEmitAssemblyAction::AnalysisOnlyAndEmitAssemblyAction(
    llvm::LLVMContext *VmContext)
    : AnalysisOnlyActionBase(Backend_EmitAssembly, VmContext) {}

void AnalysisOnlyAndEmitBCAction::anchor() {}
AnalysisOnlyAndEmitBCAction::AnalysisOnlyAndEmitBCAction(
    llvm::LLVMContext *VmContext)
    : AnalysisOnlyActionBase(Backend_EmitBC, VmContext) {}

void AnalysisOnlyAndEmitLLVMAction::anchor() {}
AnalysisOnlyAndEmitLLVMAction::AnalysisOnlyAndEmitLLVMAction(
    llvm::LLVMContext *VmContext)
    : AnalysisOnlyActionBase(Backend_EmitLL, VmContext) {}

void AnalysisOnlyAndEmitLLVMOnlyAction::anchor() {}
AnalysisOnlyAndEmitLLVMOnlyAction::AnalysisOnlyAndEmitLLVMOnlyAction(
    llvm::LLVMContext *VmContext)
    : AnalysisOnlyActionBase(Backend_EmitNothing, VmContext) {}

void AnalysisOnlyAndEmitCodeGenOnlyAction::anchor() {}
AnalysisOnlyAndEmitCodeGenOnlyAction::AnalysisOnlyAndEmitCodeGenOnlyAction(
    llvm::LLVMContext *VmContext)
    : AnalysisOnlyActionBase(Backend_EmitMCNull, VmContext) {}

void AnalysisOnlyAndEmitObjAction::anchor() {}
AnalysisOnlyAndEmitObjAction::AnalysisOnlyAndEmitObjAction(
    llvm::LLVMContext *VmContext)
    : AnalysisOnlyActionBase(Backend_EmitObj, VmContext) {}
