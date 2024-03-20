add_clang_library(MLIRCIRTransforms
  CIRFlattening.cpp
  LifetimeCheck.cpp
  LoweringPrepare.cpp
  MergeCleanups.cpp
  DropAST.cpp
  IdiomRecognizer.cpp
  LibOpt.cpp
  StdHelpers.cpp
  StructuredCFG.cpp

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
)
