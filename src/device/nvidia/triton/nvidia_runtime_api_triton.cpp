// nvidia_runtime_api_triton.cpp
#include "../../runtime_api.hpp"
#include <cuda_runtime.h>  // 使用 CUDA Runtime API

namespace llaisys::device::nvidia::triton {
namespace runtime_api {

int getDeviceCount() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

void setDevice(int device_id) {
    cudaSetDevice(device_id);
}

void deviceSynchronize() {
    cudaDeviceSynchronize();
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    cudaStreamDestroy((cudaStream_t)stream);
}

void streamSynchronize(llaisysStream_t stream) {
    cudaStreamSynchronize((cudaStream_t)stream);
}

void *mallocDevice(size_t size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void freeDevice(void *ptr) {
    cudaFree(ptr);
}

void *mallocHost(size_t size) {
    void *ptr;
    cudaMallocHost(&ptr, size);
    return ptr;
}

void freeHost(void *ptr) {
    cudaFreeHost(ptr);
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind = cudaMemcpyHostToHost;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H: cuda_kind = cudaMemcpyHostToHost; break;
        case LLAISYS_MEMCPY_H2D: cuda_kind = cudaMemcpyHostToDevice; break;
        case LLAISYS_MEMCPY_D2H: cuda_kind = cudaMemcpyDeviceToHost; break;
        case LLAISYS_MEMCPY_D2D: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default: cuda_kind = cudaMemcpyHostToHost; break;
    }
    cudaMemcpy(dst, src, size, cuda_kind);
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyKind cuda_kind = cudaMemcpyHostToHost;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H: cuda_kind = cudaMemcpyHostToHost; break;
        case LLAISYS_MEMCPY_H2D: cuda_kind = cudaMemcpyHostToDevice; break;
        case LLAISYS_MEMCPY_D2H: cuda_kind = cudaMemcpyDeviceToHost; break;
        case LLAISYS_MEMCPY_D2D: cuda_kind = cudaMemcpyDeviceToDevice; break;
        default: cuda_kind = cudaMemcpyHostToHost; break;
    }
    cudaMemcpyAsync(dst, src, size, cuda_kind, (cudaStream_t)stream);
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync
};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia::triton