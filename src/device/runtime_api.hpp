#pragma once
#include "llaisys/runtime.h"

#include "../utils.hpp"

namespace llaisys::device {
const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type);

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {
const LlaisysRuntimeAPI *getRuntimeAPI();
}

#if defined(ENABLE_NVIDIA_TRITON_API) || defined(ENABLE_NVIDIA_CUDA_API)
namespace nvidia {
#ifdef ENABLE_NVIDIA_CUDA_API
namespace cuda {
    const LlaisysRuntimeAPI *getRuntimeAPI();
} // namespace cuda
#endif
#ifdef ENABLE_NVIDIA_TRITON_API
namespace triton {
    const LlaisysRuntimeAPI *getRuntimeAPI();
} // namespace triton
#endif
} // namespace nvidia
#endif
} // namespace llaisys::device
