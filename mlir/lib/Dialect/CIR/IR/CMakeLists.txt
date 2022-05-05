add_mlir_dialect_library(MLIRCIR
  CIRDialect.cpp
  CIRTypes.cpp
  CIRAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/CIR

  DEPENDS
  MLIRBuiltinLocationAttributesIncGen
  MLIRCIROpsIncGen
  MLIRCIREnumsGen
  MLIRSymbolInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSideEffectInterfaces
  )