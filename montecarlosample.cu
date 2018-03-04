#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include "cugeometry.h"

static CudaGeometry* dev_geos = nullptr;
static int n = 0;

extern "C"
__global__ void IntersectGeo(CudaGeometry* geolist, int n, CudaRay ray)
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
  IntersectGeo<<n, 1>>(dev_geos, n, ray);
}

extern "C"
void CudaInit(CudaGeometry* geos)
{
  if(!geos)
  {
    return;
  }

  if(!dev_geos)
  {
    cudaFree(dev_geos);
  }

  n = input.size();
  for(int i=0; i<input; i++)
  {
    CudaGeometry newg;
    newg.index=i;
    newg.diffuse = CudaVec(input[i].diffuse.r, input[i].diffuse.g, input[i].diffuse.b);
    newg.emission = CudaVec(input[i].emission.r, input[i].emission.g, input[i].emission.b);
    newg.specular = CudaVec(input[i].specular.r, input[i].specular.g, input[i].specular.b);
    newg.reflectr = input[i].reflectr;
    newg.refractr = input[i].refractr;

    for(int j=0; j<input[i].vecs.size(); j++)
    {
      newg.vecs[j].p = input[i].vecs[j].p;
      newg.vecs[j].n = input[i].vecs[j].n;
      newg.vecs[j].geo = i;
    }
  }

  cudaMalloc((void**)&dev_geos, sizeof(CudaGeometry)*n);
  cudaMemcpy(dev_geos, geos, sizeof(CudaGeometry)*n, cudaMemcpyHostToDevice);
}

extern "C"
void CudaEnd()
{
  if(geos)
  {
    delete[] geos;
    geos = nullptr;
    cudaFree(dev_geos);
  }
}
