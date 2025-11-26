#include "nvidia_resource_triton.hpp"

namespace llaisys::device::nvidia::triton {
Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}
} // namespace llaisys::device::nvidia::triton    
