#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"
#include "cusgemm.h"


void cu_sgemm(float *a, float *b, float *c, int m, int n, int k) {
  cudaSetDevice(0);
  float *a_gpu, *b_gpu, *c_gpu;

  cudaMalloc( (void**) &a_gpu, m * k * sizeof(float));
  cudaMemcpy( a_gpu, a, m * k * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc( (void**) &b_gpu, n * k * sizeof(float));
  cudaMemcpy( b_gpu, b, n * k * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc( (void**) &c_gpu, m * n * sizeof(float));

  const float alpha = 1.0;
  const float beta = 0.0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k, &alpha,
              a_gpu, m,
              b_gpu, k, &beta,
              c_gpu, m);

  cudaMemcpy( c, c_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);
}

