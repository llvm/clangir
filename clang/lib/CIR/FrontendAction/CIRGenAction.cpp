//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/FrontendAction/CIRGenAction.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
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
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LLVMRemarkStreamer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTOBackend.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
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

  CompilerInstance &CI;
  DiagnosticsEngine &Diags;
  [[maybe_unused]] const HeaderSearchOptions &HeaderSearchOpts;
  CodeGenOptions &CodeGenOpts;
  [[maybe_unused]] const TargetOptions &TargetOpts;
  [[maybe_unused]] const LangOptions &LangOpts;
  const FrontendOptions &FeOptions;

  std::string InputFileName;
  std::unique_ptr<raw_pwrite_stream> OutputStream;

  ASTContext *AstContext{nullptr};
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::unique_ptr<CIRGenerator> Gen;

  /// NOTE: LinkModule is taken from clang/include/clang/CodeGen/CodeGenAction.h
  /// This is clearly suboptimal and we should reuse their functionality.
  /// Info about module to link into a module we're generating.
  struct LinkModule {
    /// The module to link in.
    std::unique_ptr<llvm::Module> Module;

    /// If true, we set attributes on Module's functions according to our
    /// CodeGenOptions and LangOptions, as though we were generating the
    /// function ourselves.
    bool PropagateAttrs;

    /// If true, we use LLVM module internalizer.
    bool Internalize;

    /// Bitwise combination of llvm::LinkerFlags used when we link the module.
    unsigned LinkFlags;
  };
  /// Bitcode modules to link in to our module.
  SmallVector<LinkModule, 4> LinkModules;

public:
  CIRGenConsumer(CIRGenAction::OutputType Action, class CompilerInstance &CI,
                 class DiagnosticsEngine &Diags,
                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                 const class HeaderSearchOptions &HeaderSearchOpts,
                 class CodeGenOptions &CodeGenOpts,
                 const class TargetOptions &TargetOpts,
                 const class LangOptions &LangOpts,
                 const FrontendOptions &FeOptions, StringRef InputFile,
                 std::unique_ptr<raw_pwrite_stream> Os)
      : Action(Action), CI(CI), Diags(Diags),
        HeaderSearchOpts(HeaderSearchOpts), CodeGenOpts(CodeGenOpts),
        TargetOpts(TargetOpts), LangOpts(LangOpts), FeOptions(FeOptions),
        InputFileName(InputFile.str()), OutputStream(std::move(Os)), FS(VFS),
        Gen(std::make_unique<CIRGenerator>(Diags, std::move(VFS),
                                           CodeGenOpts)) {}

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

  void SetupCirPipelineAndExecute(mlir::ModuleOp MlirMod,
                                  mlir::MLIRContext &MlirCtx, ASTContext &C) {
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
            MlirMod, &MlirCtx, C, !FeOptions.ClangIRDisableCIRVerifier,
            FeOptions.ClangIRLifetimeCheck, LifetimeOpts,
            FeOptions.ClangIRIdiomRecognizer, IdiomRecognizerOpts,
            FeOptions.ClangIRLibOpt, LibOptOpts, PassOptParsingFailure,
            CodeGenOpts.OptimizationLevel > 0, FlattenCir,
            !FeOptions.ClangIRDirectLowering, EnableCcLowering,
            FeOptions.ClangIREnableMem2Reg)
            .failed()) {
      if (!PassOptParsingFailure.empty()) {
        auto D = Diags.Report(diag::err_drv_cir_pass_opt_parsing);
        D << PassOptParsingFailure;
      } else
        llvm::report_fatal_error("CIR codegen: MLIR pass manager fails "
                                 "when running CIR passes!");
      return;
    }
  }

  mlir::LogicalResult writeModuleBytecode(mlir::Operation *op,
                                          raw_ostream &os) {
    // If you have a ModuleOp, pass module.getOperation().
    // writeBytecodeToFile takes an Operation* and a raw_ostream.
    if (mlir::failed(mlir::writeBytecodeToFile(op, os))) {
      return mlir::failure();
    }

    os.flush();
    return mlir::success();
  }

  void GenerateOutput(mlir::ModuleOp MlirMod,
                      std::unique_ptr<mlir::MLIRContext> MlirCtx) {
    bool EmitCIR = LangOpts.EmitCIRToFile || FeOptions.EmitClangIRFile ||
                   !LangOpts.CIRFile.empty() || !FeOptions.ClangIRFile.empty();
    if (EmitCIR) {
      std::unique_ptr<raw_pwrite_stream> CIRStream;
      llvm::SmallString<128> DefaultPath;
      if (!FeOptions.ClangIRFile.empty()) {
        CIRStream = CI.createOutputFile(FeOptions.ClangIRFile,
                                        /*Binary=*/false,
                                        /*RemoveFileOnSignal=*/true,
                                        /*UseTemporary=*/true);
      } else if (!LangOpts.CIRFile.empty()) {
        CIRStream = CI.createOutputFile(LangOpts.CIRFile,
                                        /*Binary=*/false,
                                        /*RemoveFileOnSignal=*/true,
                                        /*UseTemporary=*/true);
      } else {
        if (!FeOptions.OutputFile.empty() && FeOptions.OutputFile != "-") {
          DefaultPath = FeOptions.OutputFile;
        } else if (!InputFileName.empty() && InputFileName != "-") {
          DefaultPath = InputFileName;
        } else if (!FeOptions.Inputs.empty() && FeOptions.Inputs[0].isFile() &&
                   FeOptions.Inputs[0].getFile() != "-") {
          DefaultPath = FeOptions.Inputs[0].getFile();
        } else {
          DefaultPath = "clangir-output";
        }
        llvm::sys::path::replace_extension(DefaultPath, "cir");
        CIRStream = CI.createOutputFile(DefaultPath,
                                        /*Binary=*/false,
                                        /*RemoveFileOnSignal=*/true,
                                        /*UseTemporary=*/true);
      }

      if (CIRStream) {
        mlir::OpPrintingFlags Flags;
        Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
        MlirMod->print(*CIRStream, Flags);
      }
    }

    auto EmitMlir = [&](mlir::Operation *MlirMod, bool Verify) {
      assert(MlirMod &&
             "MLIR module does not exist, but lowering did not fail?");
      assert(OutputStream && "Why are we here without an output stream?");
      // FIXME: Think of a better error handling mechanism
      (void)writeModuleBytecode(MlirMod, *OutputStream);
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
      LlvmCtx.setDefaultTargetCPU(TargetOpts.CPU);
      LlvmCtx.setDefaultTargetFeatures(llvm::join(TargetOpts.Features, ","));

      bool DisableDebugInfo =
          CodeGenOpts.getDebugInfo() == llvm::codegenoptions::NoDebugInfo;

      LoadLinkModules(LlvmCtx);

      auto LlvmModule = lowerFromCIRToLLVMIR(
          FeOptions, MlirMod, std::move(MlirCtx), LlvmCtx,
          FeOptions.ClangIRDisableCIRVerifier,
          !FeOptions.ClangIRCallConvLowering, DisableDebugInfo);

      LlvmModule->setTargetTriple(llvm::Triple(CI.getTargetOpts().Triple));
      LlvmModule->setDataLayout(CI.getTarget().getDataLayoutString());

      LinkInModules(*LlvmModule);

      BackendAction BackendAction = getBackendActionFromOutputType(Action);

      emitBackendOutput(CI, CodeGenOpts, CI.getTarget().getDataLayoutString(),
                        LlvmModule.get(), BackendAction, FS,
                        std::move(OutputStream));
      break;
    }
    case CIRGenAction::OutputType::None:
      break;
    }
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
        SetupCirPipelineAndExecute(MlirMod, *MlirCtx, C);

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
        SetupCirPipelineAndExecute(MlirMod, *MlirCtx, C);
      }
    }

    GenerateOutput(MlirMod, std::move(MlirCtx));
  }

  void LoadLinkModules(llvm::LLVMContext &LlvmCtx) {
    for (const CodeGenOptions::BitcodeFileToLink &F :
         CI.getCodeGenOpts().LinkBitcodeFiles) {
      auto BCBuf = CI.getFileManager().getBufferForFile(F.Filename);
      if (!BCBuf) {
        CI.getDiagnostics().Report(diag::err_cannot_open_file)
            << F.Filename << BCBuf.getError().message();
        LinkModules.clear();
        return;
      }

      Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
          getOwningLazyBitcodeModule(std::move(*BCBuf), LlvmCtx);
      if (!ModuleOrErr) {
        handleAllErrors(ModuleOrErr.takeError(), [&](llvm::ErrorInfoBase &EIB) {
          CI.getDiagnostics().Report(diag::err_cannot_open_file)
              << F.Filename << EIB.message();
        });
        LinkModules.clear();
        return;
      }
      LinkModules.push_back({std::move(ModuleOrErr.get()), F.PropagateAttrs,
                             F.Internalize, F.LinkFlags});
    }
    return;
  }

  // Links each entry in LinkModules into our module. Returns true on error.
  void LinkInModules(llvm::Module &M) {
    llvm::Linker L(M);

    for (auto &LM : LinkModules) {
      // Link the module using LLVM's linker
      if (llvm::Linker::linkModules(M, std::move(LM.Module), LM.LinkFlags)) {
        CI.getDiagnostics().Report(diag::err_fe_linking_module)
            << M.getModuleIdentifier() << LM.Module->getModuleIdentifier();
        return;
      }
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
      Ci.getLangOpts(), Ci.getFrontendOpts(), InputFile, std::move(Out));
  cgConsumer = Result.get();

  // Enable generating macro debug info only when debug info is not disabled and
  // also macrod ebug info is enabled
  if (Ci.getCodeGenOpts().getDebugInfo() != llvm::codegenoptions::NoDebugInfo &&
      Ci.getCodeGenOpts().MacroDebugInfo) {
    llvm_unreachable("NYI");
  }

  return std::move(Result);
}

static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
loadModule(std::unique_ptr<llvm::MemoryBuffer> buf,
           mlir::MLIRContext &mlirContext) {
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(buf), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &mlirContext);
  if (!module)
    return mlir::failure();
  return module;
}

void CIRGenAction::ExecuteAction() {
  if (getCurrentFileKind().getLanguage() != Language::CIR) {
    this->ASTFrontendAction::ExecuteAction();
    return;
  }

  auto &Ci = getCompilerInstance();
  auto &Diags = Ci.getDiagnostics();
  const clang::FrontendOptions &Fo = Ci.getFrontendOpts();

  if (Fo.Inputs.size() > 1)
    llvm_unreachable("NYI: Missing support of 'linking CIR files'");

  const FrontendInputFile &Input = Fo.Inputs.front();
  StringRef InputFile = Input.getFile();
  InputKind Kind = Input.getKind();
  assert(Kind.getFormat() == InputKind::Source &&
         "Loading CIR files only support source code formats");
  auto Out = Ci.takeOutputStream();
  auto &SourceManager = Ci.getSourceManager();
  auto FileId = SourceManager.getMainFileID();

  if (!Out)
    Out = getOutputStream(Ci, InputFile, action);

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      action, Ci, Ci.getDiagnostics(), &Ci.getVirtualFileSystem(),
      Ci.getHeaderSearchOpts(), Ci.getCodeGenOpts(), Ci.getTargetOpts(),
      Ci.getLangOpts(), Ci.getFrontendOpts(), InputFile, std::move(Out));
  cgConsumer = Result.get();

  std::unique_ptr<mlir::MLIRContext> MlirContext{new mlir::MLIRContext};
  MlirContext->getOrLoadDialect<mlir::DLTIDialect>();
  MlirContext->getOrLoadDialect<mlir::func::FuncDialect>();
  MlirContext->getOrLoadDialect<cir::CIRDialect>();
  MlirContext->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  MlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();
  MlirContext->getOrLoadDialect<mlir::omp::OpenMPDialect>();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> InputOrErr =
      Ci.getFileManager().getBufferForFile(InputFile);
  if (!InputOrErr) {
    std::error_code Ec = InputOrErr.getError();
    Diags.Report(clang::diag::err_fe_error_reading) << InputFile;
    Diags.Report(clang::diag::note_drv_command_failed_diag_msg) << Ec.message();
    return;
  }
  std::unique_ptr<llvm::MemoryBuffer> InputBuf = std::move(*InputOrErr);

  auto MlirModuleOr = loadModule(std::move(InputBuf), *MlirContext);

  if (mlir::failed(MlirModuleOr)) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << "failed to parse CIR module" << InputFile;
    return;
  }

  // FIXME: This introduces a leak. The ownership model of "GenerateOutput" is
  // puzzling. We give ownership of MLIRContext, but not of the MLIRModule.
  // If the lifetime of the ModuleOp exceeds the lifetime of the context there
  // are crashes.
  // I need to check this with CIR team.
  mlir::ModuleOp MlirModule = std::move(*MlirModuleOr).release();

  assert(MlirModule && "Could not load module");
  cgConsumer->GenerateOutput(std::move(MlirModule), std::move(MlirContext));
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
      Ci.getTargetOpts(), Ci.getLangOpts(), Ci.getFrontendOpts(), InFile,
      nullptr));
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
