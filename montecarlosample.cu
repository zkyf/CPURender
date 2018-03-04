#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include "cugeometry.h"

struct CudaVec;
struct CudaVec4;
struct CudaVertex;
struct CudaGeometry;
struct CudaRay;

struct CudaVec
{
  double x, y, z;

  CUDA_CALLABLE_MEMBER CudaVec(double _x=0.0, double _y=0.0, double _z=0.0) : x(_x), y(_y), z(_z) {}

  CUDA_CALLABLE_MEMBER double Dot(const CudaVec& b) { return x*b.x+y*b.y+z*b.z; }
  CUDA_CALLABLE_MEMBER CudaVec Cross(const CudaVec& b) const { return CudaVec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
  CUDA_CALLABLE_MEMBER double Length() const { return sqrt(x*x+y*y+z*z); }
  CUDA_CALLABLE_MEMBER CudaVec Normalized() const { return CudaVec(x/Length(), y/Length(), z/Length()); }
  CUDA_CALLABLE_MEMBER void Print() const { printf("CudaVec(%lf, %lf, %lf)", x, y, z); }
};

struct CudaVec4
{
  double x, y, z, w;

  CUDA_CALLABLE_MEMBER CudaVec4(double _x=0.0, double _y=0.0, double _z=0.0, double _w=0.0) : x(_x), y(_y), z(_z), w(_w) {}
  CUDA_CALLABLE_MEMBER CudaVec4(CudaVec v3, double _w=0.0) : x(v3.x), y(v3.y), z(v3.z), w(_w) {}
  CUDA_CALLABLE_MEMBER double Dot(const CudaVec4& b) { return x*b.x+y*b.y+z*b.z+w*b.w; }
  CUDA_CALLABLE_MEMBER double Length() const { return sqrt(x*x+y*y+z*z+w*w); }
  CUDA_CALLABLE_MEMBER CudaVec4 Normalized() const { return CudaVec4(x/Length(), y/Length(), z/Length(), w/Length()); }
  CUDA_CALLABLE_MEMBER CudaVec Vec3() const { return CudaVec(x, y, z); }
  CUDA_CALLABLE_MEMBER void Print() const { printf("CudaVe4c(%lf, %lf, %lf, %lf)", x, y, z, w); }
};

CUDA_CALLABLE_MEMBER CudaVec operator+(const CudaVec& a, const CudaVec& b)    { return CudaVec(a.x+b.x, a.y+b.y, a.z+b.z); }
CUDA_CALLABLE_MEMBER CudaVec operator-(const CudaVec& a, const CudaVec& b)    { return CudaVec(a.x+b.x, a.y+b.y, a.z+b.z); }
CUDA_CALLABLE_MEMBER CudaVec operator*(const CudaVec& a, const double& b)     { return CudaVec(a.x*b, a.y*b, a.z*b); }
CUDA_CALLABLE_MEMBER CudaVec operator*(const double& b, const CudaVec& a)     { return CudaVec(a.x*b, a.y*b, a.z*b); }
CUDA_CALLABLE_MEMBER CudaVec operator/(const CudaVec& a, const double& b)     { return CudaVec(a.x/b, a.y/b, a.z/b); }
CUDA_CALLABLE_MEMBER CudaVec operator/(const double& b, const CudaVec& a)     { return CudaVec(a.x/b, a.y/b, a.z/b); }

CUDA_CALLABLE_MEMBER CudaVec4 operator+(const CudaVec4& a, const CudaVec4& b) { return CudaVec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
CUDA_CALLABLE_MEMBER CudaVec4 operator-(const CudaVec4& a, const CudaVec4& b) { return CudaVec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w-b.w); }
CUDA_CALLABLE_MEMBER CudaVec4 operator*(const CudaVec4& a, const double& b)   { return CudaVec4(a.x*b, a.y*b, a.z*b, a.w*b); }
CUDA_CALLABLE_MEMBER CudaVec4 operator*(const double& b, const CudaVec4& a)   { return CudaVec4(a.x*b, a.y*b, a.z*b, a.w*b); }
CUDA_CALLABLE_MEMBER CudaVec4 operator/(const CudaVec4& a, const double& b)   { return CudaVec4(a.x/b, a.y/b, a.z/b, a.w/b); }
CUDA_CALLABLE_MEMBER CudaVec4 operator/(const double& b, const CudaVec4& a)   { return CudaVec4(a.x/b, a.y/b, a.z/b, a.w/b); }

struct CudaVertex
{
  CudaVec4 p;
  CudaVec4 n;
  bool valid = true;
  int geo;

  CUDA_CALLABLE_MEMBER CudaVertex(bool v=true) : valid(v) {}
  CUDA_CALLABLE_MEMBER static CudaVertex Intersect(CudaVertex a, CudaVertex b, double r)
  {
    CudaVertex result;
    result.n = a.n*r+b.n*(1-r);
    result.p = a.p*r+b.p*(1-r);

    result.valid = a.valid&&b.valid;
    result.geo = a.geo;
  }
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
  CUDA_CALLABLE_MEMBER CudaVec Normal() const
  {
    return (vecs[2].p-vecs[1].p).Vec3().Cross(vecs[0].p.Vec3()-vecs[1].p.Vec3()).Normalized();
  }

  CUDA_CALLABLE_MEMBER CudaVec4 Plane() const
  {
    CudaVec4 n(Normal(), 0.0);
    n.w = (n.Dot(vecs[0].p));
    return n;
  }
};

struct CudaRay
{
  CudaVec4 o;
  CudaVec4 n;

  CUDA_CALLABLE_MEMBER CudaVertex IntersectGeo(const CudaGeometry& geo) const
  {
    printf("CudaRay IntersectGeo # %d: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf)\n", geo.index, o.x, o.y, o.z, n.x, n.y, n.z);
    return CudaVertex(false);
//    CudaVec4 pi = geo.Plane();
//    double pn = pi.Dot(n);
//    if(fabs(pn)<1e-3)
//    {
//      printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), parallel geo # %d  pi=(%lf, %lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, pi.x, pi.y, pi.z, pi.w);
//      return CudaVertex(false);
//    }
//    double r = -pi.Dot(o)/pn;
//    if(r<0) return CudaVertex(false);
//    CudaVec4 p=r*n+o; // intersection point with the plane
//    printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), hit geo # %d @ p=(%lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, p.x, p.y, p.z);

//    if((geo.vecs[0].p-p).Vec3().Length()<1e-3)
//    {
//      return geo.vecs[0];
//    }

//    CudaVec4 cp = p-geo.vecs[0].p;
//    cp.w =0;
//    CudaVec4 cpn = cp.Normalized();
//    CudaVec4 ca = geo.vecs[1].p - geo.vecs[0].p;
//    CudaVec4 cb = geo.vecs[2].p - geo.vecs[0].p;
//    CudaVec4 cd = ca.Dot(cpn)*cpn;
//    CudaVec4 ce = cb.Dot(cpn)*cpn;
//    double rb = (ca-cd).Length()/(cb-ce).Length();
//    if((ca-cd).Dot(cb-ce)>0) rb=-rb;
//    rb = rb/(1+rb);
//    double ra = 1-rb;
//    CudaVec4 f = rb*geo.vecs[2].p+ra*geo.vecs[1].p;
//    double rc = 1-cp.Length()/(f-geo.vecs[0].p).Length();
//    if(cp.Dot(f-geo.vecs[0].p)<0) rc=-1;

//    if(ra<0 || rb<0 || rc<0 || ra>1 || rb>1 || rc>1) return CudaVertex(false);
//    else
//    {
//      CudaVertex vf = CudaVertex::Intersect(geo.vecs[1], geo.vecs[2], ra);
//      CudaVertex vp = CudaVertex::Intersect(geo.vecs[0], vf, rc);
//      vp.geo = geo.index;
//      vp.valid=true;
//  //    if((p-vp.p).length()>1e-3) return VertexInfo(false);
//  //    qDebug() << "p=" << p << On(p) << "p-p'=" << (p-vp.p).length() << "ra=" << ra << "rb=" << rb << "rc=" << rc;
//      return vp;
//    }
  }

  CUDA_CALLABLE_MEMBER void Print() const { printf("Ray: o="); o.Print(); printf(", n="); n.Print(); }
};

static CudaGeometry* dev_geos = nullptr;
static int n = 0;
static FILE* cudaOutput;
static bool* hits = nullptr;
static bool* dev_hits = nullptr;

extern "C"
__global__ void IntersectGeo(CudaGeometry* geolist, int n, const CudaRay& ray, bool* dev_hits)
{
  printf("\n");
  printf("CUDA block (%d, %d, %d) thread (%d %d %d) : ray\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  if(blockIdx.x<n)
  {
    CudaVertex hp = ray.IntersectGeo(geolist[blockIdx.x]);
    ray.Print();
    printf("\n");
    if(hp.valid)
    {
      printf("CUDA # %d : hit geo with index # %d\n", blockIdx.x, geolist[blockIdx.x].index);
      dev_hits[blockIdx.x]=true;
    }
    else
    {
      dev_hits[blockIdx.x]=false;
    }
  }
  printf("\n");
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
  printf("into cuda part n=%d\n", n);
  fflush(stdout);
  IntersectGeo<<<n, 1>>>(dev_geos, n, ray, dev_hits);
  hits = new bool[n];
  cudaMemcpy(hits, dev_hits, sizeof(bool) * n, cudaMemcpyDeviceToHost);
  for(int i=0; i<n; i++)
  {
    if(hits[i])
    {
      printf("geo #%d hit\n", i);
    }
    else
    {
      printf("geo #%d NOT hit\n", i);
    }
  }
  fflush(stdout);
  delete[] hits;
}

extern "C"
void CudaInit(CudaGeometry* geos, int _n)
{
//  cudaOutput = fopen("cudaoutput.txt", "w");
  printf("cuda part init %d\n", _n);
  fflush(stdout);
  if(!geos)
  {
    return;
  }

  if(!dev_geos)
  {
    cudaFree(dev_geos);
    cudaFree(dev_hits);
  }

  n = _n;

  cudaMalloc((void**)&dev_geos, sizeof(CudaGeometry)*n);
  cudaMalloc((void**)&dev_hits, sizeof(bool)*n);
  cudaMemcpy(dev_geos, geos, sizeof(CudaGeometry)*n, cudaMemcpyHostToDevice);
}

extern "C"
void CudaEnd()
{
  if(dev_geos)
  {
    cudaFree(dev_geos);
    cudaFree(dev_hits);
  }
}
