add_mlir_dialect(CIROps cir)
add_mlir_doc(CIROps CIROps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS CIROps.td)
mlir_tablegen(CIROpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(CIROpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CIROpsStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(CIROpsStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRCIREnumsGen)