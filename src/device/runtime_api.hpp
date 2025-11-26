#pragma once
#include "llaisys/runtime.h"

#include "../utils.hpp"

namespace llaisys::device {
const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type);

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {
const LlaisysRuntimeAPI *getRuntimeAPI();
}

namespace nvidia {
namespace cuda {
    const LlaisysRuntimeAPI *getRuntimeAPI();
} // namespace cuda
namespace triton {
    const LlaisysRuntimeAPI *getRuntimeAPI();
} // namespace triton

} // namespace nvidia

} // namespace llaisys::device
