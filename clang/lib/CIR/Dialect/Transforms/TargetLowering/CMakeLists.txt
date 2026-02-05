add_clang_library(MLIRCIRTargetLowering
  ABIInfo.cpp
  ABIInfoImpl.cpp
  CIRCXXABI.cpp
  CIRLowerContext.cpp
  CIRRecordLayout.cpp
  ItaniumCXXABI.cpp
  LowerCall.cpp
  LowerFunction.cpp
  LowerModule.cpp
  LowerTypes.cpp
  RecordLayoutBuilder.cpp
  TargetInfo.cpp
  TargetLoweringInfo.cpp
  Targets/AArch64.cpp
  Targets/AMDGPU.cpp
  Targets/NVPTX.cpp
  Targets/SPIR.cpp
  Targets/X86.cpp

  DEPENDS
  clangBasic

  LINK_COMPONENTS
  TargetParser

  LINK_LIBS PUBLIC

  clangBasic
  MLIRIR
  MLIRPass
  MLIRDLTIDialect
  MLIRCIR
  MLIRCIRInterfaces
  MLIRLLVMDialect
)
