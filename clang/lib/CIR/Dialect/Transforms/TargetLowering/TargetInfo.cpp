#include "TargetLoweringInfo.h"

namespace mlir {
namespace cir {

TargetLoweringInfo::TargetLoweringInfo(std::unique_ptr<ABIInfo> info)
    : Info(std::move(info)) {}

TargetLoweringInfo::~TargetLoweringInfo() = default;

} // namespace cir
} // namespace mlir
