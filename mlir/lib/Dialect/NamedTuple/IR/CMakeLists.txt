add_mlir_dialect_library(MLIRNamedTupleDialect
  NamedTupleDialect.cpp
  NamedTupleOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/NamedTuple

  DEPENDS
  MLIRNamedTupleIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
