#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
//#include "cugeometry.h"

#define M_PI (3.1415926)

struct CudaVec;
struct CudaVec4;
struct CudaVertex;
struct CudaGeometry;
struct CudaRay;

struct CudaVec
{
  double x, y, z;

  __host__ __device__ CudaVec(double _x=0.0, double _y=0.0, double _z=0.0) : x(_x), y(_y), z(_z) {}

  __host__ __device__ double Dot(const CudaVec& b) { return x*b.x+y*b.y+z*b.z; }
  __host__ __device__ CudaVec Cross(const CudaVec& b) const { return CudaVec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
  __host__ __device__ double Length() const { return sqrt(x*x+y*y+z*z); }
  __host__ __device__ CudaVec Normalized() const { return CudaVec(x/Length(), y/Length(), z/Length()); }
  __host__ __device__ void Print() const { printf("CudaVec(%lf, %lf, %lf)", x, y, z); }
};

struct CudaVec4
{
  double x, y, z, w;

  __host__ __device__ CudaVec4(double _x=0.0, double _y=0.0, double _z=0.0, double _w=0.0) : x(_x), y(_y), z(_z), w(_w) {}
  __host__ __device__ CudaVec4(CudaVec v3, double _w=0.0) : x(v3.x), y(v3.y), z(v3.z), w(_w) {}
  __host__ __device__ double Dot(const CudaVec4& b) { return x*b.x+y*b.y+z*b.z+w*b.w; }
  __host__ __device__ double Length() const { return sqrt(x*x+y*y+z*z+w*w); }
  __host__ __device__ CudaVec4 Normalized() const { return CudaVec4(x/Length(), y/Length(), z/Length(), w/Length()); }
  __host__ __device__ CudaVec Vec3() const { return CudaVec(x, y, z); }
  __host__ __device__ void Print() const { printf("CudaVe4c(%lf, %lf, %lf, %lf)", x, y, z, w); }
};

__host__ __device__ CudaVec operator+(const CudaVec& a, const CudaVec& b)    { return CudaVec(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ CudaVec operator-(const CudaVec& a, const CudaVec& b)    { return CudaVec(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ CudaVec operator*(const CudaVec& a, const double& b)     { return CudaVec(a.x*b, a.y*b, a.z*b); }
__host__ __device__ CudaVec operator*(const double& b, const CudaVec& a)     { return CudaVec(a.x*b, a.y*b, a.z*b); }
__host__ __device__ CudaVec operator/(const CudaVec& a, const double& b)     { return CudaVec(a.x/b, a.y/b, a.z/b); }
__host__ __device__ CudaVec operator/(const double& b, const CudaVec& a)     { return CudaVec(a.x/b, a.y/b, a.z/b); }

__host__ __device__ CudaVec4 operator+(const CudaVec4& a, const CudaVec4& b) { return CudaVec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
__host__ __device__ CudaVec4 operator-(const CudaVec4& a, const CudaVec4& b) { return CudaVec4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
__host__ __device__ CudaVec4 operator*(const CudaVec4& a, const double& b)   { return CudaVec4(a.x*b, a.y*b, a.z*b, a.w*b); }
__host__ __device__ CudaVec4 operator*(const double& b, const CudaVec4& a)   { return CudaVec4(a.x*b, a.y*b, a.z*b, a.w*b); }
__host__ __device__ CudaVec4 operator/(const CudaVec4& a, const double& b)   { return CudaVec4(a.x/b, a.y/b, a.z/b, a.w/b); }
__host__ __device__ CudaVec4 operator/(const double& b, const CudaVec4& a)   { return CudaVec4(a.x/b, a.y/b, a.z/b, a.w/b); }

struct CudaVertex
{
  CudaVec4 p;
  CudaVec4 n;
  bool valid = true;
  int geo;

  __host__ __device__ CudaVertex(bool v=true) : valid(v) {}
  __host__ __device__ static CudaVertex Intersect(CudaVertex a, CudaVertex b, double r)
  {
    CudaVertex result;
    result.n = a.n*r+b.n*(1-r);
    result.p = a.p*r+b.p*(1-r);

    result.valid = a.valid&&b.valid;
    result.geo = a.geo;

    return result;
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
  __host__ __device__ CudaVec Normal() const
  {
    return (vecs[2].p-vecs[1].p).Vec3().Cross(vecs[0].p.Vec3()-vecs[1].p.Vec3()).Normalized();
  }

  __host__ __device__ CudaVec4 Plane() const
  {
    CudaVec4 n(Normal(), 0.0);
    n.w = (-n.Dot(vecs[0].p));
    return n;
  }
  __host__ __device__ void Print() const
  {
    printf("CudaGeometry # %d\n", index);
    for(int i=0; i<3; i++)
    {
      printf("v#%d p=", i); vecs[i].p.Print();
      printf(", n="); vecs[i].n.Print();
      printf("\n");
    }
  }
};

struct CudaRay
{
  CudaVec4 o;
  CudaVec4 n;

  __host__ __device__ CudaVertex IntersectGeo(CudaGeometry geo) const
  {
//    printf("CudaRay IntersectGeo # %d: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf)\n", geo.index, o.x, o.y, o.z, n.x, n.y, n.z);
//    return CudaVertex(false);
//    geo.Print();
    CudaVec4 pi = geo.Plane();
//    printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), geo # %d  pi=(%lf, %lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, pi.x, pi.y, pi.z, pi.w);
    double pn = pi.Dot(n);
    if(fabs(pn)<1e-3)
    {
//      printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), parallel geo # %d  pi=(%lf, %lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, pi.x, pi.y, pi.z, pi.w);
      return CudaVertex(false);
    }
    double r = -pi.Dot(o)/pn;
//    printf("r=%lf, pn=%lf\n", r, pn);
    if(r<0) return CudaVertex(false);
    CudaVec4 p=r*n+o; // intersection point with the plane
//    printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), hit geo # %d @ p=(%lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, p.x, p.y, p.z);

    if((geo.vecs[0].p-p).Vec3().Length()<1e-3)
    {
      return geo.vecs[0];
    }

    CudaVec4 cp = p-geo.vecs[0].p;
    cp.w =0;
    CudaVec4 cpn = cp.Normalized();
    CudaVec4 ca = geo.vecs[1].p - geo.vecs[0].p;
    CudaVec4 cb = geo.vecs[2].p - geo.vecs[0].p;
    CudaVec4 cd = ca.Dot(cpn)*cpn;
    CudaVec4 ce = cb.Dot(cpn)*cpn;

//    printf("cp="); cp.Print(); printf("\n");
//    printf("ca="); ca.Print(); printf("\n");
//    printf("cb="); cb.Print(); printf("\n");
//    printf("cd="); cd.Print(); printf("\n");
//    printf("ce="); ce.Print(); printf("\n");

    double rb = (ca-cd).Length()/(cb-ce).Length();
    if((ca-cd).Dot(cb-ce)>0) rb=-rb;
    rb = rb/(1+rb);
    double ra = 1-rb;
    CudaVec4 f = rb*geo.vecs[2].p+ra*geo.vecs[1].p;
    double rc = 1-cp.Length()/(f-geo.vecs[0].p).Length();
    if(cp.Dot(f-geo.vecs[0].p)<0) rc=-1;

//    printf("ra=%lf rb=%lf rc=%lf\n", ra, rb, rc);

    if(ra<0 || rb<0 || rc<0 || ra>1 || rb>1 || rc>1) return CudaVertex(false);
    else
    {
      CudaVertex vf = CudaVertex::Intersect(geo.vecs[1], geo.vecs[2], ra);
      CudaVertex vp = CudaVertex::Intersect(geo.vecs[0], vf, rc);
      vp.geo = geo.index;
      vp.valid=true;
  //    if((p-vp.p).length()>1e-3) return VertexInfo(false);
  //    qDebug() << "p=" << p << On(p) << "p-p'=" << (p-vp.p).length() << "ra=" << ra << "rb=" << rb << "rc=" << rc;
      return vp;
    }
  }

  __host__ __device__ void Print() const { printf("CudaRay: o=(%lf, %lf, %lf) , n=(%lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z); }
};

static CudaGeometry* dev_geos = nullptr;
static int n = 0;
static FILE* cudaOutput;
static double* hits = nullptr;
static double* dev_hits = nullptr;

__global__ void IntersectGeo(CudaGeometry* geolist, int n, CudaRay ray, double* dev_hits)
{
//  printf("\n");
//  printf("CUDA block (%d, %d, %d) thread (%d %d %d) : ray\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  if(blockIdx.x<n)
  {
    CudaVertex hp = ray.IntersectGeo(geolist[blockIdx.x]);
//    ray.Print();
//    printf("\n");
    if(hp.valid)
    {
//      printf("CUDA # %d : hit geo with index # %d\n", blockIdx.x, geolist[blockIdx.x].index);
      dev_hits[blockIdx.x]=(hp.p-ray.o).Vec3().Length();
    }
    else
    {
//      printf("CUDA # %d : NOT hit geo with index # %d\n", blockIdx.x, geolist[blockIdx.x].index);
      dev_hits[blockIdx.x]=-1;
    }
  }
//  printf("\n");
}

extern "C"
void CudaIntersect(CudaRay ray)
{
//  printf("into cuda part n=%d\n", n);
//  ray.Print();
  fflush(stdout);
  IntersectGeo<<<n, 1>>>(dev_geos, n, ray, dev_hits);
  cudaMemcpy(hits, dev_hits, sizeof(double) * n, cudaMemcpyDeviceToHost);
  for(int i=0; i<n; i++)
  {
    if(hits[i] >=0 )
    {
//      printf("geo #%d hit\n", i);
    }
    else
    {
//      printf("geo #%d NOT hit\n", i);
    }
  }
//  printf("cuda part ended");
  fflush(stdout);
}

extern "C"
void CudaInit(CudaGeometry* geos, int _n, double* h)
{
//  cudaOutput = fopen("cudaoutput.txt", "w");
  hits=h;
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
  cudaMalloc((void**)&dev_hits, sizeof(double)*n);
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

__device__ CudaVec CudaMonteCarloSample(CudaVertex o, CudaRay i, int depth)
{

}

__device__ CudaRay GetRay(int xx, int yy, int width, int height, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right)
{
  float vh = M_PI/3;
  float vw = vh/height * width;
  float vhp = vh/height;
  float vwp = vw/width;

  CudaRay ray;
  ray.o = CudaVec4(camera, 1.0);
  ray.n = CudaVec4((forward+right*tan(vwp*xx-vw/2)+up*tan(vhp*(height - yy - 1)-vh/2)).Normalized(), 0.0);

  return ray;
}

__global__ void CudaMonteCarloRender(CudaGeometry* geolist, int n, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer)
{
  int xx=blockIdx.x;
  int yy=blockIdx.y;
  if(xx>=w || yy>=h) return;
  int index = xx+yy*w;
  CudaRay ray = GetRay(xx, yy, w, h, camera, up, forward, right);

  for(int i=0; i<n; i++)
  {
    CudaVertex hp = ray.IntersectGeo(geolist[i]);
    if(hp.valid)
    {
      buffer[index] = geolist[hp.geo].diffuse;
    }
  }

//  printf("CudaMonteCarloRender : x=%d, y=%d\n", xx, yy);

//  buffer[index] = (ray.n.Vec3()+CudaVec(1.0, 1.0, 1.0))/2.0;
}

extern "C" void CudaRender(int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer)
{
  CudaVec* dev_buffer;
  cudaMalloc((void**)&dev_buffer, sizeof(CudaVec)*w*h);

  CudaMonteCarloRender<<<dim3(w, h), 1>>>(dev_geos, n, w, h, camera, up, forward, right, dev_buffer);

  cudaMemcpy(buffer, dev_buffer, sizeof(CudaVec)*w*h, cudaMemcpyDeviceToHost);
}
