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

static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);

void bitReverse(Complex *data); 
__global__ void cooley_tukey_fft(Complex *d); 
void myFft(Complex *d);
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
  bitReverse(h_signal);
  // Allocate device memory for signal
  Complex *d_signal;
  cudaMalloc(reinterpret_cast<void **>(&d_signal), mem_size);
  // Copy host memory to device
  cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice);
  // ! Kernel Call
  myFft(d_signal);
  // ! Result Check
  cudaMemcpy(h_signal, d_signal, mem_size,cudaMemcpyDeviceToHost);
  for (unsigned int i = 0; i < N; ++i) {
    printf("The %dth element:(x: %f, y: %f)\n",i,h_signal[i].x,h_signal[i].y);
  }
  // ! Clean up Memory
  free(h_signal);
  cudaFree(d_signal);

  return 0;
}

void bitReverse(Complex *data) {
  // In-Place bit reverse
  for (unsigned int i = 0; i < N; i++)
  {
    int rev_i = 0;
    for (unsigned int j = 0; j < log2f(N); j++) {
      rev_i = (rev_i << 1) | ((i >> j) & 1);
    }
    if (rev_i > i)
    {
      Complex temp = data[i];
      data[i] = data[rev_i];
      data[rev_i] = temp;
    }
  }
  // Check Correctness
  
  /*
  for (int i = 0;i < N; ++i) {
    printf("%f\n",data[i].x);
  }

  */
  
}
void myFft(Complex *d) {
  printf("myFFT is Running...\n");
  // ! Cooley-Tukey Framework
  cooley_tukey_fft<<<1,256>>>(d);
}
__global__ void cooley_tukey_fft(Complex *d) {
  int tId = threadIdx.x + blockDim.x * blockIdx.x;
  int stride;
  float unit_angle;
  Complex unit_root;
  int idx;
  Complex twiddle;
  for (int stage = 1; stage <= log2f(N); stage++) {
    stride = 1 << stage;
    idx = tId * stride;
    if (idx < N) {
      unit_angle = -2.0f * M_PI / (float) stride; 
      unit_root = {cos(unit_angle),sin(unit_angle)};
      twiddle = {1.0f,0.0f};
      for (int i = 0; i < stride / 2; i++) {
        Complex t = ComplexMul(twiddle,d[idx + i + stride / 2]);
        Complex u = d[idx + i];
        d[idx + i] = ComplexAdd(u,t);
        d[idx + i + stride / 2] = ComplexAdd(u,ComplexScale(t,-1));
        twiddle = ComplexMul(twiddle,unit_root);
      }
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations: Borrowed from CudaSamples
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}
