set(deps)
if (NOT CLANG_BUILT_STANDALONE)
  set(deps
    MLIRBuiltinLocationAttributesIncGen
    MLIRSymbolInterfacesIncGen
    )
endif()

add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIROpenCLAttrs.cpp
  CIRDataLayout.cpp
  CIRDialect.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  FPEnv.cpp

  DEPENDS
  ${deps}
  MLIRCIROpsIncGen
  MLIRCIREnumsGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen

  LINK_LIBS PUBLIC
  clangAST
  )

mlir_target_link_libraries(MLIRCIR PUBLIC
  MLIRIR
  MLIRCIRInterfaces
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRLLVMDialect
  MLIRSideEffectInterfaces)
