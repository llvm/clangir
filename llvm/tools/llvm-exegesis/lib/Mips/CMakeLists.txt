include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/Mips
  ${LLVM_BINARY_DIR}/lib/Target/Mips
  )

set(LLVM_LINK_COMPONENTS
  Mips
  Exegesis
  Core
  Support
  TargetParser
  MC
  )

add_llvm_library(LLVMExegesisMips
  DISABLE_LLVM_LINK_LLVM_DYLIB
  STATIC
  Target.cpp

  DEPENDS
  intrinsics_gen
  MipsCommonTableGen
  )
