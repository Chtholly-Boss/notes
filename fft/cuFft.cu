
// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// includes, project
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

// Complex data type
typedef float2 Complex;
void cuFft(Complex *d,Complex *h);
#define N 8

int main(int argc,char** argv) {
  // ! Allocate Memory
  int mem_size = sizeof(Complex) * N;
  // Allocate Host Memory for signal
  Complex * h_signal = 
    reinterpret_cast<Complex *> (malloc(mem_size)); 
  // Initialize the memory for the signal
  for (unsigned int i = 0; i < N; ++i) {
  h_signal[i].x = i;
  //h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
  h_signal[i].y = 0;
  }
  // Allocate device memory for signal
  Complex *d_signal;
  cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size);
  // Copy host memory to device
  cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice);
  // ! Kernel Call
  cuFft(d_signal,h_signal);
  // ! Result Check
  //cudaMemcpy(h_signal, d_signal, mem_size,cudaMemcpyDeviceToHost);
  for (unsigned int i = 0; i < N; ++i) {
    printf("The %dth element:(x: %f, y: %f)\n",i,h_signal[i].x,h_signal[i].y);
  }
  // ! Clean up Memory
  free(h_signal);
  cudaFree(d_signal);

  return 0;
}

void cuFft(Complex *d,Complex *h) {
  printf("cuFFT is Running...\n");
  cufftHandle plan;
  cufftPlan1d(&plan,N,CUFFT_C2C,1);

  printf("Transforming Signal Using cuFFT\n");
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d),
                              reinterpret_cast<cufftComplex *>(h),
                              CUFFT_FORWARD);
  cufftDestroy(plan);
}