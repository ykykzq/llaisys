#include "nvidia_resource.cuh"

namespace llaisys::device::nvidia::cuda {
Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}
} // namespace llaisys::device::nvidia::cuda    
