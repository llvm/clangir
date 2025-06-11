set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(link_libs
  clangBasic
  clangCodeGen
  clangDriver
  clangExtractAPI
  clangFrontend
  clangRewriteFrontend
  )

set(deps)

if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND link_libs
    clangStaticAnalyzerFrontend
    )
endif()

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  ClangDriverOptions
  ${deps}

  LINK_LIBS
  ${link_libs}
  )
