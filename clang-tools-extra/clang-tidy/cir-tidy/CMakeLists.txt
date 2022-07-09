set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )
include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

add_subdirectory(tool)

add_clang_library(CIRTidy
    CIRTidy.cpp
    CIRASTConsumer.cpp

    LINK_LIBS
    clangTidy

    DEPENDS
    omp_gen
)

clang_target_link_libraries(CIRTidy
  PRIVATE
  clangBasic
  clangTooling
  clangToolingCore
  )
