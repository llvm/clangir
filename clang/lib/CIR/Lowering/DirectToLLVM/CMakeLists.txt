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

add_clang_library(clangCIRLoweringDirectToLLVM
  LowerToLLVMIR.cpp
  LowerToLLVM.cpp

  DEPENDS
  MLIRCIREnumsGen
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
  clangCIRLoweringHelpers
  MLIRCIR
  )
mlir_target_link_libraries(clangCIRLoweringDirectToLLVM PUBLIC
  ${dialect_libs}
  MLIRAnalysis
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRCIRTransforms
  MLIRIR
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRSupport
  MLIRMemRefDialect
  MLIROpenMPDialect
  MLIROpenMPToLLVMIRTranslation
  )

target_include_directories(clangCIRLoweringDirectToLLVM PRIVATE
  ${CLANG_SOURCE_DIR}/lib/CIR/Dialect/Transforms/TargetLowering
  )
