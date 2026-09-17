// Hello-world CUDA kernel: vector addition
// This validates the full pipeline: nvcc compile → PTX load → cudarc launch → result readback

extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
