#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include "cugeometry.h"

static CudaGeometry* dev_geos = nullptr;
static int n = 0;
static FILE* cudaOutput;

extern "C"
__global__ void IntersectGeo(CudaGeometry* geolist, int n, const CudaRay& ray)
{
  if(blockIdx.x<n)
  {
    CudaVertex hp = ray.IntersectGeo(geolist[blockIdx.x]);
    if(hp.valid)
    {
      printf("CUDA # %d : hit geo with index # %d\n", blockIdx.x, geolist[blockIdx.x].index);
    }
  }
}

extern "C"
__global__ void testKernel(int val)
{
    printf("[%d, %d]:\t\tValue is:%d\n",\
            blockIdx.y*gridDim.x+blockIdx.x,\
            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
            val);
}

extern "C"
void CudaIntersect(const CudaRay& ray)
{
  printf("into cuda part\n");
  IntersectGeo<<<n, 1>>>(dev_geos, n, ray);
}

extern "C"
void CudaInit(CudaGeometry* geos, int _n)
{
  cudaOutput = fopen("cudaoutput.txt", "w");
  fprintf(cudaOutput, "cuda part init %d\n", _n);
  if(!geos)
  {
    return;
  }

  if(!dev_geos)
  {
    cudaFree(dev_geos);
  }

  n = _n;

  cudaMalloc((void**)&dev_geos, sizeof(CudaGeometry)*n);
  cudaMemcpy(dev_geos, geos, sizeof(CudaGeometry)*n, cudaMemcpyHostToDevice);
}

extern "C"
void CudaEnd()
{
  if(dev_geos)
  {
    cudaFree(dev_geos);
  }
}
