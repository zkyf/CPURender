#ifndef CUGEOMETRY_H
#define CUGEOMETRY_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define CUDA_CALLABLE_MEMBER
#include <math.h>
#include "ray.h"

struct CudaVec;
struct CudaVec4;
struct CudaVertex;
struct CudaGeometry;
struct CudaRay;

struct CudaVec
{
  double x, y, z;

  CUDA_CALLABLE_MEMBER CudaVec(double _x=0.0, double _y=0.0, double _z=0.0) : x(_x), y(_y), z(_z) {}
  CUDA_CALLABLE_MEMBER CudaVec(QVector3D a) : x(a.x()), y(a.y()), z(a.z()) {}
};

struct CudaVec4
{
  double x, y, z, w;

  CUDA_CALLABLE_MEMBER CudaVec4(double _x=0.0, double _y=0.0, double _z=0.0, double _w=0.0) : x(_x), y(_y), z(_z), w(_w) {}
  CUDA_CALLABLE_MEMBER CudaVec4(CudaVec v3, double _w=0.0) : x(v3.x), y(v3.y), z(v3.z), w(_w) {}
};


struct CudaVertex
{
  CudaVec4 p;
  CudaVec4 n;
  bool valid = true;
  int geo;

  CUDA_CALLABLE_MEMBER CudaVertex(bool v=true) : valid(v) {}
};

struct CudaGeometry
{
  int index;

  CudaVertex vecs[3];
  CudaVec diffuse = CudaVec(1.0, 1.0, 1.0);
  CudaVec specular = CudaVec();
  CudaVec emission = CudaVec();
  double reflectr = 0.0;
  double refractr = 0.0;
};

struct CudaRay
{
  CudaVec4 o;
  CudaVec4 n;

  void FromRay(const Ray& i)
  {
    n = CudaVec4(i.n.x(), i.n.y(), i.n.z(), 0.0);
    o = CudaVec4(i.o.x(), i.o.y(), i.o.z(), 1.0);
  }
};

#endif

#endif // CUGEOMETRY_H
