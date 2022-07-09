#include "CIRASTConsumer.h"
#include "clang/CIR/Dialect/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace cir;

namespace clang {

CIRASTConsumer::CIRASTConsumer(CompilerInstance &CI, StringRef inputFile) {
  Gen =
      std::make_unique<CIRGenerator>(CI.getDiagnostics(), CI.getCodeGenOpts());
}

bool CIRASTConsumer::HandleTopLevelDecl(DeclGroupRef D) {
  PrettyStackTraceDecl CrashInfo(*D.begin(), SourceLocation(),
                                 AstContext->getSourceManager(),
                                 "CIR generation of declaration");
  Gen->HandleTopLevelDecl(D);
  return true;
}

void CIRASTConsumer::Initialize(ASTContext &Context) {
  AstContext = &Context;
  Gen->Initialize(Context);
}

void CIRASTConsumer::HandleTranslationUnit(ASTContext &C) {
  Gen->HandleTranslationUnit(C);
  Gen->verifyModule();

  mlir::ModuleOp mlirMod = Gen->getModule();
  std::unique_ptr<mlir::MLIRContext> mlirCtx = Gen->takeContext();

  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(/*prettyForm=*/false);

  SourceManager &SourceMgr = C.getSourceManager();
  FileID MainFileID = SourceMgr.getMainFileID();

  llvm::MemoryBufferRef MainFileBuf = SourceMgr.getBufferOrFake(MainFileID);
  std::unique_ptr<llvm::MemoryBuffer> FileBuf =
      llvm::MemoryBuffer::getMemBuffer(MainFileBuf);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(FileBuf), llvm::SMLoc());

  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, mlirCtx.get());

  mlir::PassManager pm(mlirCtx.get());
  pm.addPass(mlir::createMergeCleanupsPass());
  pm.addPass(mlir::createLifetimeCheckPass());

  bool Result = !mlir::failed(pm.run(mlirMod));
  if (!Result)
    llvm::report_fatal_error(
        "The pass manager failed to run pass on the module!");
}
} // namespace clang
