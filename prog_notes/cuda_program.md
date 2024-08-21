# Cuda Programming
### Configuration
`nvidia-smi` to see which version of cuda to be installed
Go to [cuda-toolkit-Download-Site](https://developer.nvidia.com/cuda-toolkit) to download the corresponding version

Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.
![grid-of-clusters](./figure/cuda/grid-of-clusters.png)
Thread blocks that belong to a cluster have access to the Distributed Shared Memory. 
![memory-hierarchy](./figure/cuda/memory-hierarchy.png)
the CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C++ program. This is the case, for example, when the kernels execute on a GPU and the rest of the C++ program executes on a CPU.
The CUDA programming model also assumes that both the host and the device maintain their own separate memory spaces in DRAM, referred to as host memory and device memory, respectively. 

The compute capability of a device is represented by a version number, also sometimes called its “SM version”. This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.

The compute capability comprises a major revision number X and a minor revision number Y and is denoted by X.Y.
type `nvidia-smi --query-gpu=compute_cap --format=csv` to get your compute capability
> On my Lenovo,it is 6.1,which is the Pascal architecture

### Programming Interface {#interface}
`nvcc` is a compiler driver that simplifies the process of compiling C++ or PTX code: It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages.

Binary code is architecture-specific. A cubin object is generated using the compiler option `-code` that specifies the targeted architecture: For example, compiling with `-code=sm_80` produces binary code for devices of compute capability 8.0. Binary compatibility is guaranteed from one minor revision to the next one, but not from one minor revision to the previous one or across major revisions. In other words, a cubin object generated for compute capability X.y will only execute on devices of compute capability X.z where z≥y.

Linear memory can also be allocated through `cudaMallocPitch()` and `cudaMalloc3D()`. These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in Device Memory Accesses, therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the `cudaMemcpy2D()` and `cudaMemcpy3D()` functions). The returned pitch (or stride) must be used to access array elements.

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```
```c++
// access global variables
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```
`cudaGetSymbolAddress()` is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. The size of the allocated memory is obtained through `cudaGetSymbolSize()`.
### API
#### cuFFT 
Self-Optimization for FFT see [My FFT Notes](./fft.md).This section is just for describing the use of cuFFT,lay the foundation of `How to Profiler the performance` Section.

To call cuFFT routines in `foo.cu`,include file `cufft.h` or `cufftXt.h` into `foo.cu` and include the library in the link line like the following:

`nvcc [options] filename.cu … -I/usr/local/cuda/inc -L/usr/local/cuda/lib -lcufft`

### Profile
#### Docs
[Nsight Systems](https://docs.nvidia.com/nsight-systems/index.html)
[Nsight Compute](https://docs.nvidia.com/nsight-compute/index.html)
#### Notes
##### Nsight System
`nsys --help` to get common uses
- `nsys profile foo`: specify options and begin analysis.
