add_subdirectory(TargetLowering)

add_clang_library(MLIRCIRTransforms
  ABILowering.cpp
  CallConvLowering.cpp
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  CXXABILowering.cpp
  FlattenCFG.cpp
  HoistAllocas.cpp
  LoweringPrepare.cpp
  LoweringPrepareItaniumCXXABI.cpp
  GotoSolver.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic

  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRInterfaces
  MLIRCIRTargetLowering
)
