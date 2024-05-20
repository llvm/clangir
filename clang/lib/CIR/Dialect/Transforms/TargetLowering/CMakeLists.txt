add_clang_library(TargetLowering
  ABIInfo.cpp
  ABIInfoImpl.cpp
  CIRCXXABI.cpp
  CIRContext.cpp
  CIRRecordLayout.cpp
  DataLayout.cpp
  ItaniumCXXABI.cpp
  LowerCall.cpp
  LowerFunction.cpp
  LowerModule.cpp
  LowerTypes.cpp
  RecordLayoutBuilder.cpp
  TargetInfo.cpp
  TargetLoweringInfo.cpp
  Targets/X86.cpp

  DEPENDS

  LINK_LIBS PUBLIC

  MLIRIR
  MLIRPass

  MLIRCIR
  MLIRCIRInterfaces
)
