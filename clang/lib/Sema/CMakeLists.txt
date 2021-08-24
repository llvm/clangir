set(LLVM_LINK_COMPONENTS
  Core
  Demangle
  FrontendHLSL
  FrontendOpenMP
  MC
  Support
  TargetParser
  )

clang_tablegen(OpenCLBuiltins.inc -gen-clang-opencl-builtins
  SOURCE OpenCLBuiltins.td
  TARGET ClangOpenCLBuiltinsImpl
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangSema
  AnalysisBasedWarnings.cpp
  CodeCompleteConsumer.cpp
  CIRBasedWarnings.cpp
  DeclSpec.cpp
  DelayedDiagnostic.cpp
  HLSLExternalSemaSource.cpp
  IdentifierResolver.cpp
  JumpDiagnostics.cpp
  MultiplexExternalSemaSource.cpp
  ParsedAttr.cpp
  Scope.cpp
  ScopeInfo.cpp
  Sema.cpp
  SemaAccess.cpp
  SemaAttr.cpp
  SemaAPINotes.cpp
  SemaAvailability.cpp
  SemaBase.cpp
  SemaCXXScopeSpec.cpp
  SemaCast.cpp
  SemaChecking.cpp
  SemaCodeComplete.cpp
  SemaConcept.cpp
  SemaConsumer.cpp
  SemaCoroutine.cpp
  SemaCUDA.cpp
  SemaDecl.cpp
  SemaDeclAttr.cpp
  SemaDeclCXX.cpp
  SemaDeclObjC.cpp
  SemaExceptionSpec.cpp
  SemaExpr.cpp
  SemaExprCXX.cpp
  SemaExprMember.cpp
  SemaExprObjC.cpp
  SemaFixItUtils.cpp
  SemaHLSL.cpp
  SemaInit.cpp
  SemaLambda.cpp
  SemaLookup.cpp
  SemaModule.cpp
  SemaObjCProperty.cpp
  SemaOpenACC.cpp
  SemaOpenMP.cpp
  SemaOverload.cpp
  SemaPseudoObject.cpp
  SemaRISCVVectorLookup.cpp
  SemaStmt.cpp
  SemaStmtAsm.cpp
  SemaStmtAttr.cpp
  SemaSYCL.cpp
  SemaTemplate.cpp
  SemaTemplateDeduction.cpp
  SemaTemplateInstantiate.cpp
  SemaTemplateInstantiateDecl.cpp
  SemaTemplateVariadic.cpp
  SemaType.cpp
  TypeLocBuilder.cpp

  DEPENDS
  ClangOpenCLBuiltinsImpl
  omp_gen
  ClangDriverOptions
  MLIRCIROpsIncGen

  LINK_LIBS
  clangAPINotes
  clangAST
  clangAnalysis
  clangBasic
  clangEdit
  clangLex
  clangSupport

  clangCIR
  ${dialect_libs}
  MLIRCIR
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  )
