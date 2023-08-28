add_clang_library(MLIRCIRASTAttrInterfaces
  ASTAttrInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIRASTAttrInterfacesIncGen

  LINK_LIBS
  clangCIR
  ${dialect_libs}
  MLIRCIR
  MLIRAnalysis
  MLIRIR
  MLIRSupport
 )
