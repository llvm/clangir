if(LIBC_TARGET_OS_IS_BAREMETAL)
  return()
endif()

add_libc_tool_unittest(
  wrappergen_test
  SRCS 
    wrappergen_test.cpp   
  ARGS
    --path=${LIBC_SOURCE_DIR}
    --tool=${LLVM_BINARY_DIR}/bin/libc-wrappergen
    --api=${LIBC_SOURCE_DIR}/test/utils/tools/WrapperGen/testapi.td
)

add_dependencies(wrappergen_test libc-wrappergen)
