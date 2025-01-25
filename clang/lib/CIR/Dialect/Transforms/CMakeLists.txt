add_subdirectory(TargetLowering)

add_clang_library(MLIRCIRTransforms
  LifetimeCheck.cpp
  LoweringPrepare.cpp
  CIRCanonicalize.cpp
  CIRSimplify.cpp
  DropAST.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  StdHelpers.cpp
  FlattenCFG.cpp
  GotoSolver.cpp
  SCFPrepare.cpp
  CallConvLowering.cpp
  HoistAllocas.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  clangAST
  clangBasic
  TargetLowering

  MLIRCIR
  MLIRCIRInterfaces
)

mlir_target_link_libraries(MLIRCIRTransforms PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
