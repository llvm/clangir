set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRBuilder.cpp
  CIRGenTypes.cpp
  CIRRecordLayoutBuilder.cpp
  LowerToLLVM.cpp

  DEPENDS
  MLIRCIROpsIncGen
  MLIRSubElementInterfacesIncGen

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  ${dialect_libs}
  MLIRCIR
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRef
  MLIRMemRefToLLVM
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSCFToStandard
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
  )
