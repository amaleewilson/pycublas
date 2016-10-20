#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cblas.h>

#include "cublas_v2.h"
#include "cusgemm.h"




/*#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  if (abort) exit(code);
  }
} */



void cu_sgemm(float *a, float *b, float *c, int m, int n, int k) {

//  float *a_gpu, *b_gpu, *c_gpu;

  //gpuErrchk(
//  cudaMalloc( (void**) &a_gpu, m * n * sizeof(float));
  //);
  //gpuErrchk(
//  cudaMemcpy( a_gpu, a, m * n * sizeof(float), cudaMemcpyHostToDevice);
  //);

  //gpuErrchk(
//  cudaMalloc( (void**) &b_gpu, n * k * sizeof(float));
  //);

  //gpuErrchk(
//  cudaMemcpy( b_gpu, b, n * k * sizeof(float), cudaMemcpyHostToDevice);
  //);

  //gpuErrchk(
//  cudaMalloc( (void**) &c_gpu, m * k * sizeof(float));
  //);

  const float alpha = 1.0;
  const float beta = 0.0;

/*  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k, &alpha,
              a_gpu, m,
              b_gpu, k, &beta,
              c_gpu, m); */

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
//  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k, 1.0,
              a, m,
              b, k, 0.0,
              c, m);

  //gpuErrchk(
//  cudaMemcpy( c, c_gpu, m * k * sizeof(float), cudaMemcpyDeviceToHost);
  //);
}

