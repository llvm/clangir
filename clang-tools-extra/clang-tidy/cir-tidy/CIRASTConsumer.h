#include "clang/AST/ASTContext.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/Frontend/CompilerInstance.h"

namespace clang {
class CIRASTConsumer : public ASTConsumer {
public:
  CIRASTConsumer(CompilerInstance &CI, StringRef inputFile);

private:
  void Initialize(ASTContext &Context) override;
  void HandleTranslationUnit(ASTContext &C) override;
  bool HandleTopLevelDecl(DeclGroupRef D) override;
  std::unique_ptr<cir::CIRGenerator> Gen;
  ASTContext *AstContext{nullptr};
};
} // namespace clang
