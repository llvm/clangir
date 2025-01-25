set(
  LLVM_LINK_COMPONENTS
  Core
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(deps)
if (NOT CLANG_BUILT_STANDALONE)
  set(deps ${dialect_libs})
endif()

add_clang_library(clangCIR
  CIRAsm.cpp
  CIRGenAtomic.cpp
  CIRGenBuilder.cpp
  CIRGenBuiltin.cpp
  CIRGenBuiltinAArch64.cpp
  CIRGenBuiltinX86.cpp
  CIRGenCXX.cpp
  CIRGenCXXABI.cpp
  CIRGenCall.cpp
  CIRGenClass.cpp
  CIRGenCleanup.cpp
  CIRGenCoroutine.cpp
  CIRGenDecl.cpp
  CIRGenDeclCXX.cpp
  CIRGenException.cpp
  CIRGenExpr.cpp
  CIRGenExprComplex.cpp
  CIRGenExprConst.cpp
  CIRGenExprAgg.cpp
  CIRGenExprCXX.cpp
  CIRGenExprScalar.cpp
  CIRGenFunction.cpp
  CIRGenItaniumCXXABI.cpp
  CIRGenModule.cpp
  CIRGenOpenCLRuntime.cpp
  CIRGenOpenCL.cpp
  CIRGenOpenMPRuntime.cpp
  CIRGenPointerAuth.cpp
  CIRGenStmt.cpp
  CIRGenStmtOpenMP.cpp
  CIRGenTBAA.cpp
  CIRGenTypes.cpp
  CIRGenVTables.cpp
  CIRGenerator.cpp
  CIRPasses.cpp
  CIRRecordLayoutBuilder.cpp
  ConstantInitBuilder.cpp
  TargetInfo.cpp

  DEPENDS
  MLIRCIR
  MLIRCIROpsIncGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen
  ${deps}

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  MLIRCIR
  MLIRCIRTransforms
  MLIRCIRInterfaces
)

mlir_target_link_libraries(clangCIR PUBLIC
  ${dialect_libs}
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRDLTIDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRParser
  MLIRPass
  MLIRSCFToControlFlow
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTargetLLVMIRImport
  MLIRTargetLLVMIRExport
  MLIRTransforms
)
