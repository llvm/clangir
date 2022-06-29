# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES CIRTidyMain.cpp CIRTidyToolMain.cpp)

add_clang_library(CIRTidyMain
  CIRTidyMain.cpp
  )

add_clang_executable(cir-tidy
  CIRTidyToolMain.cpp
  )

target_link_libraries(cir-tidy
  PRIVATE
  CIRTidyMain
  )

install(TARGETS cir-tidy
  DESTINATION bin
  )
