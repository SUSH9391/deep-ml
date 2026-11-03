#include <cuda_runtime.h>
#include <vector>

__global__ void kernel(const float* in, float* out, int N) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to load multiple elements per thread if N > blockDim.x * gridDim.x
    float sum = 0.0f;
    while (i < N) {
        sum += in[i];
        i += blockDim.x * gridDim.x;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = min(64, (N + threadsPerBlock - 1) / threadsPerBlock);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, blocksPerGrid * sizeof(float));
    
    cudaMemcpy(d_in, input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_in, d_out, N);
    
    std::vector<float> h_out(blocksPerGrid);
    cudaMemcpy(h_out.data(), d_out, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Final reduction on CPU for block results
    float total_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; ++i) {
        total_sum += h_out[i];
    }
    
    *const_cast<float*>(output) = total_sum;
    
    cudaFree(d_in);
    cudaFree(d_out);
}