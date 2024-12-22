set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(deps)
if (NOT CLANG_BUILT_STANDALONE)
  set(deps
    MLIRBuiltinLocationAttributesIncGen
    MLIRBuiltinTypeInterfacesIncGen
    MLIRFunctionInterfacesIncGen
    )
endif()

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen
  ${deps}

  LINK_LIBS
  clangAST
  clangBasic
  clangCodeGen
  clangLex
  clangFrontend
  clangCIR
  clangCIRLoweringDirectToLLVM
  clangCIRLoweringThroughMLIR
  MLIRCIR
  )

mlir_target_link_libraries(clangCIRFrontendAction PUBLIC
  ${dialect_libs}
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  )
