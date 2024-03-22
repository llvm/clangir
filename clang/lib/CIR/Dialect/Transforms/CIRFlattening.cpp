#include "mlir/Pass/PassManager.h"
#include "clang/CIR/Dialect/Passes.h"

void mlir::populateCIRFlatteningPasses(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createFlattenCFGPass());
}