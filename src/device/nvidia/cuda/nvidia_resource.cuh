#pragma once

#include "../../device_resource.hpp"

namespace llaisys::device::nvidia {
namespace cuda {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};
} // namespace cuda
} // namespace llaisys::device::nvidia
