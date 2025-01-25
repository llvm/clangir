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

add_clang_library(clangCIRLoweringHelpers
  LoweringHelpers.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRCIREnumsGen
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
  MLIRCIR
  )

mlir_target_link_libraries(clangCIRLoweringHelpers PUBLIC
  ${dialect_libs}
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  )

add_subdirectory(DirectToLLVM)
add_subdirectory(ThroughMLIR)
