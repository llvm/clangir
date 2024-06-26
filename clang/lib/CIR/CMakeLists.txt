include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

add_compile_options("-Werror=deprecated-declarations")

add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Lowering)
add_subdirectory(Interfaces)
