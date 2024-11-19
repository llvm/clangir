#include "../ClangTidyDiagnosticConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/Frontend/CompilerInstance.h"

using namespace clang;

namespace cir {
namespace tidy {
class CIRASTConsumer : public ASTConsumer {
public:
  CIRASTConsumer(CompilerInstance &CI, StringRef inputFile,
                 clang::tidy::ClangTidyContext &Context);

private:
  void Initialize(ASTContext &Context) override;
  void HandleTranslationUnit(ASTContext &C) override;
  bool HandleTopLevelDecl(DeclGroupRef D) override;
  std::unique_ptr<CIRGenerator> Gen;
  ASTContext *AstContext{nullptr};
  clang::tidy::ClangTidyContext &Context;
};
} // namespace tidy
} // namespace cir
