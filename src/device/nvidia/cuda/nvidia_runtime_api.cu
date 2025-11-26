#include "../../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::nvidia::cuda {
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
    switch(kind){
        case LLAISYS_MEMCPY_H2H: cudaMemcpy(dst, src, size, cudaMemcpyHostToHost); break;
        case LLAISYS_MEMCPY_H2D: cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice); break;
        case LLAISYS_MEMCPY_D2H: cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost); break;
        case LLAISYS_MEMCPY_D2D: cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice); break;
        default: cudaMemcpy(dst, src, size, cudaMemcpyHostToHost); break;
    }
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    switch(kind){
        case LLAISYS_MEMCPY_H2H: cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, (cudaStream_t)stream); break;
        case LLAISYS_MEMCPY_H2D: cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream); break;
        case LLAISYS_MEMCPY_D2H: cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream); break;
        case LLAISYS_MEMCPY_D2D: cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream); break;
        default: cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost,(cudaStream_t)stream); break;
    }
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
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia::cuda
