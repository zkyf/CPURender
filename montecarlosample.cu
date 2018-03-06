#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
//#include "cugeometry.h"

#define M_PI (3.1415926)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct CudaVec;
struct CudaVec4;
struct CudaVertex;
struct CudaGeometry;
struct CudaRay;

struct CudaVec
{
  double x, y, z;

  __host__ __device__ CudaVec(double _x=0.0, double _y=0.0, double _z=0.0) : x(_x), y(_y), z(_z) {}

  __host__ __device__ double Dot(const CudaVec& b)    const { return x*b.x+y*b.y+z*b.z; }
  __host__ __device__ CudaVec Cross(const CudaVec& b) const { return CudaVec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
  __host__ __device__ double Length()                 const { return sqrt(x*x+y*y+z*z); }
  __host__ __device__ CudaVec Normalized()            const { return CudaVec(x/Length(), y/Length(), z/Length()); }
  __host__ __device__ void Print()                    const { printf("CudaVec(%lf, %lf, %lf)", x, y, z); }

  __host__ __device__ void operator=(const CudaVec& b) { x=b.x; y=b.y; z=b.z; }
};

struct CudaVec4
{
  double x, y, z, w;

  __host__ __device__ CudaVec4(double _x=0.0, double _y=0.0, double _z=0.0, double _w=0.0) : x(_x), y(_y), z(_z), w(_w) {}
  __host__ __device__ CudaVec4(CudaVec v3, double _w=0.0) : x(v3.x), y(v3.y), z(v3.z), w(_w) {}

  __host__ __device__ double   Dot(const CudaVec4& b) const { return x*b.x+y*b.y+z*b.z+w*b.w; }
  __host__ __device__ double   Length()               const { return sqrt(x*x+y*y+z*z+w*w); }
  __host__ __device__ CudaVec4 Normalized()           const { return CudaVec4(x/Length(), y/Length(), z/Length(), w/Length()); }
  __host__ __device__ CudaVec  Vec3()                 const { return CudaVec(x, y, z); }
  __host__ __device__ void     Print()                const { printf("CudaVe4c(%lf, %lf, %lf, %lf)", x, y, z, w); }
};

__host__ __device__ CudaVec operator+(const CudaVec& a, const CudaVec& b)    { return CudaVec(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ CudaVec operator-(const CudaVec& a, const CudaVec& b)    { return CudaVec(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ CudaVec operator*(const CudaVec& a, const CudaVec& b)    { return CudaVec(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ CudaVec operator*(const CudaVec& a, const double& b)     { return CudaVec(a.x*b, a.y*b, a.z*b); }
__host__ __device__ CudaVec operator*(const double& b, const CudaVec& a)     { return CudaVec(a.x*b, a.y*b, a.z*b); }
__host__ __device__ CudaVec operator/(const CudaVec& a, const double& b)     { return CudaVec(a.x/b, a.y/b, a.z/b); }

__host__ __device__ CudaVec4 operator+(const CudaVec4& a, const CudaVec4& b) { return CudaVec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
__host__ __device__ CudaVec4 operator-(const CudaVec4& a, const CudaVec4& b) { return CudaVec4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
__host__ __device__ CudaVec4 operator*(const CudaVec4& a, const double& b)   { return CudaVec4(a.x*b, a.y*b, a.z*b, a.w*b); }
__host__ __device__ CudaVec4 operator*(const double& b, const CudaVec4& a)   { return CudaVec4(a.x*b, a.y*b, a.z*b, a.w*b); }
__host__ __device__ CudaVec4 operator/(const CudaVec4& a, const double& b)   { return CudaVec4(a.x/b, a.y/b, a.z/b, a.w/b); }

struct CudaVertex
{
  CudaVec4 p;
  CudaVec4 n;
  bool valid;
  int geo;

  __host__ __device__ CudaVertex(bool v=true) : valid(v), p(), n(), geo(-1) {}
  __host__ __device__ static CudaVertex Intersect(const CudaVertex& a, const CudaVertex& b, double r)
  {
    CudaVertex result;
    result.n = a.n*r+b.n*(1-r);
    result.p = a.p*r+b.p*(1-r);

    result.valid = a.valid&&b.valid;
    result.geo = a.geo;

    return result;
  }
  __host__ __device__ void operator=(const CudaVertex& b)
  {
    p=b.p;
    n=b.n;
    valid=b.valid;
    geo=b.geo;
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

  __host__ __device__ void IntersectGeo(const CudaGeometry& geo, CudaVertex& result) const
  {
//    printf("CudaRay IntersectGeo # %d: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf)\n", geo.index, o.x, o.y, o.z, n.x, n.y, n.z);
//    geo.Print();
    CudaVec4 pi = geo.Plane();
    const CudaVertex c = geo.vecs[0];
    const CudaVertex a = geo.vecs[1];
    const CudaVertex b = geo.vecs[2];

//    printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), geo # %d  pi=(%lf, %lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, pi.x, pi.y, pi.z, pi.w);
    double pn = pi.Dot(n);

    if(fabs(pn)<1e-3)
    {
//      printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), parallel geo # %d  pi=(%lf, %lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, pi.x, pi.y, pi.z, pi.w);
      result.valid=false;
      return;
    }

    double r = -pi.Dot(o)/pn;

//    printf("r=%lf, pn=%lf\n", r, pn);
    if(r<0)
    {
      result.valid=false;
      return;
    }
    CudaVec4 p=r*n+o; // intersection point with the plane

//    printf("CudaRay IntersectGeo: o=(%lf, %lf, %lf), n=(%lf, %lf, %lf), hit geo # %d @ p=(%lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z, geo.index, p.x, p.y, p.z);

    if((c.p-p).Vec3().Length()<1e-3)
    {
      result.valid=false;
      return;
    }

    CudaVec4 cp = p-c.p;
    cp.w =0;
    CudaVec4 cpn = cp.Normalized();
    CudaVec4 ca = a.p - c.p;
    CudaVec4 cb = b.p - c.p;
    CudaVec4 cd = ca.Dot(cpn)*cpn;
    CudaVec4 ce = cb.Dot(cpn)*cpn;

//    printf("cp="); cp.Print(); printf("\n");
//    printf("ca="); ca.Print(); printf("\n");
//    printf("cb="); cb.Print(); printf("\n");
//    printf("cd="); cd.Print(); printf("\n");
//    printf("ce="); ce.Print(); printf("\n");

    if((ca-cd).Dot(cb-ce)>0)
    {
      result.valid=false;
      return;
    }
    double rb, ra;
    if((cb-ce).Length()>1e-3)
    {
      rb = (ca-cd).Length()/(cb-ce).Length();
      if(rb<0 || rb>1)
      {
        result.valid=false;
        return;
      }
      rb = rb/(1+rb);
      ra = 1-rb;
    }
    else
    {
      ra = 0;
      rb = 1.0;
    }
    CudaVec4 f = rb*b.p+ra*a.p;
    if((f-c.p).Length()<1e-3)
    {
      result.valid=false;
      return;
    }
    double rc = 1-cp.Length()/(f-c.p).Length();
    if(cp.Dot(f-c.p)<0)
    {
      result.valid=false;
      return;
    }

//    printf("ra=%lf rb=%lf rc=%lf\n", ra, rb, rc);

    if(ra<0 || rb<0 || rc<0 || ra>1 || rb>1 || rc>1)
    {
      result.valid=false;
      return;
    }
    else
    {
      CudaVertex vf = CudaVertex::Intersect(a, b, ra);
      CudaVertex vp = CudaVertex::Intersect(c, vf, rc);
      vp.geo = geo.index;
      vp.valid=true;
      result = vp;
    }
  }

  __host__ __device__ void Print() const { printf("CudaRay: o=(%lf, %lf, %lf) , n=(%lf, %lf, %lf)\n", o.x, o.y, o.z, n.x, n.y, n.z); }
};

static CudaGeometry* dev_geos = nullptr;
static int n = 0;
static FILE* cudaOutput;
static double* hits = nullptr;
static double* dev_hits = nullptr;
static double* dev_randNum = nullptr;
static int randN = 0;

extern "C"
void CudaIntersect(CudaRay ray)
{
//  printf("into cuda part n=%d\n", n);
//  ray.Print();
  fflush(stdout);
//  IntersectGeo<<<n, 1>>>(dev_geos, n, ray, dev_hits);
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
void CudaInit(CudaGeometry* geos, int _n, double* h, double* randNum, int _randN)
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
    gpuErrchk(cudaFree(dev_geos));
    gpuErrchk(cudaFree(dev_hits));
    gpuErrchk(cudaFree(dev_randNum))
  }

  n = _n;
  randN = _randN;

  printf("CUDA: sizeof(CudaGeometry)=%d, sizeof(CudaVec)=%d, n=%d\n", sizeof(CudaGeometry), sizeof(CudaVec), n);

  gpuErrchk( cudaMalloc((void**)&dev_geos, sizeof(CudaGeometry)*n));
  printf("cudaMalloc((void**)&dev_geos, sizeof(CudaGeometry)*n)\n");

  gpuErrchk( cudaMalloc((void**)&dev_hits, sizeof(double)*n));
  printf("cudaMalloc((void**)&dev_hits, sizeof(double)*n)\n");

  gpuErrchk( cudaMalloc((void**)&dev_randNum, sizeof(double)*randN));
  printf("cudaMalloc((void**)&dev_randNum, sizeof(double)*randN)\n");

  gpuErrchk( cudaMemcpy(dev_geos, geos, sizeof(CudaGeometry)*n, cudaMemcpyHostToDevice));
  printf("cudaMemcpy(dev_geos, geos, sizeof(CudaGeometry)*n, cudaMemcpyHostToDevice)\n");

  gpuErrchk( cudaMemcpy(dev_randNum, randNum, sizeof(double)*randN, cudaMemcpyHostToDevice));
  printf("cudaMemcpy(dev_randNum, randNum, sizeof(double)*randN, cudaMemcpyHostToDevice)\n");

  gpuErrchk( cudaPeekAtLastError() );
  fflush(stdout);
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

extern "C"
__device__ void CudaMonteCarloSample(CudaGeometry* geolist, int n, CudaVertex o, CudaRay i, double* randNum, int randN, CudaVec& result)
{
//  result = (i.n.Vec3()+CudaVec(1.0, 1.0, 1.0))/2;
  return;

//  if(!o.valid) return;

//  CudaVec currentColor=geolist[o.geo].diffuse;
//  if(o.n.Dot(i.n),0) o.n = -1.0*o.n;

//  bool haslight=false;
//  for(int level=0; level<5; level++)
//  {
//    if(!o.valid) break;
//    float sin2theta=randNum[level]; // sin^2(theta)
//    float sintheta=sqrt(sin2theta);
//    float phi = randNum[level*19]*2*M_PI;
//    CudaRay ray;
//    ray.o=o.p;
//    CudaVec w = o.n.Vec3();
//    CudaVec u = (fabs(w.x)>0.1?CudaVec(0, 1):CudaVec(1)).Cross(w);
//    CudaVec v = w.Cross(u);
//    ray.n = CudaVec4(sintheta*cos(phi)*u+v*sintheta*sin(phi)+w*sqrt(1-sin2theta), 0);

//    double mind=1e20;
//    CudaVertex minp(false);

//    for(int i=0; i<n; i++)
//    {
//      CudaVertex hp;
//      ray.IntersectGeo(geolist[i], hp);
//      if(hp.valid)
//      {
//        if((hp.p-o.p).Length()>1e-3 && (hp.p-o.p).Length()<mind)
//        {
//          mind=(hp.p-o.p).Length();
//          minp=hp;
//        }
//      }
//    }

//    if(minp.valid)
//    {
//      currentColor = currentColor * geolist[minp.geo].diffuse + currentColor * geolist[minp.geo].emission;
//      if(geolist[minp.geo].emission.Length()>0.1)
//      {
//        haslight=true;
//      }
//    }

//    o = minp;
//  }
//  if(haslight)
//  {
//    result = currentColor;
//    return;
//  }
//  else
//  {
//    return;
//  }
}

extern "C"
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

extern "C"
__global__ void debug_GetRay(int xx, int yy, int width, int height, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaRay* result)
{
  printf("debug_GetRay\n");
  float vh = M_PI/3;
  float vw = vh/height * width;
  float vhp = vh/height;
  float vwp = vw/width;

  CudaRay ray;
  ray.o = CudaVec4(camera, 1.0);
  ray.n = CudaVec4((forward+right*tan(vwp*xx-vw/2)+up*tan(vhp*(height - yy - 1)-vh/2)).Normalized(), 0.0);

  printf("ray.n=(%lf, %lf, %lf)\n", ray.n.x, ray.n.y, ray.n.z);
  *result = ray;
}

#define SampleNum (32)

extern "C"
__global__ void CudaMonteCarloRender(CudaGeometry* geolist, int n, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer, double* randNum, int randN)
{
  if(n==0) return;

  int xx=blockIdx.x;
  int yy=blockIdx.y;
  if(xx<0 || xx>=w || yy<0 || yy>=h) return;
  int index = xx+yy*gridDim.x;
  CudaRay ray = GetRay(xx, yy, w, h, camera, up, forward, right);
  buffer[index] = CudaVec(0, 0, 0);

  for(int sp=0; sp<SampleNum; sp++)
  {
    double mind=1e20;
    CudaVertex minp(false);

    buffer[index] = buffer[index]+(ray.n.Vec3()+CudaVec(1.0, 1.0, 1.0))/2;

    for(int gi=0; gi<n; gi++)
    {
      CudaVertex hp(false);
      CudaGeometry geo = geolist[gi];
      ray.IntersectGeo(geo, hp);
      if(hp.valid==true)
      {
        double d=(hp.p.Vec3()-camera).Length();
        if(d>1e-3 && d<mind)
        {
          mind=d;
          minp=hp;
        }
      }
    }

    if(minp.valid)
    {
//      minp.valid=false;
      CudaVec result;
      CudaMonteCarloSample(geolist, n, minp, ray, randNum, randN, result);
      buffer[index] = CudaVec(0.0, 0.0, 0.0);
//      buffer[index] = buffer[index] + result;
    }
  }

  buffer[index] = buffer[index]/SampleNum;

  return;
}

extern "C"
__global__ void CudaDivide(int w, int h, CudaVec* buffer)
{
  int xx=blockIdx.x;
  int yy=blockIdx.y;
  if(xx<0 || xx>=w || yy<0 || yy>=h) return;
  int index = xx+yy*gridDim.x;
  buffer[index] = buffer[index]/4;
}

extern "C" void CudaRender(int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer)
{
  CudaVec* dev_buffer;
  memset(buffer, 0, sizeof(CudaVec)*w*h);
  gpuErrchk( cudaMalloc((void**)&dev_buffer, sizeof(CudaVec)*w*h));

  printf("start cuda render\n");
  printf("buffer.size=%d %d\n", sizeof(buffer), sizeof(buffer)/sizeof(CudaVec));
  fflush(stdout);
//  for(int i=0; i<4; i++)
//  {
    CudaMonteCarloRender<<<dim3(w, h), 1>>>(dev_geos, n, w, h, camera, up, forward, right, dev_buffer, dev_randNum, randN);
//  }
//  CudaDivide<<<dim3(w, h), 1>>>(w, h, dev_buffer);
  fflush(stdout);
  printf("end cuda render\n");
  fflush(stdout);

//  gpuErrchk( cudaPeekAtLastError() );
//  gpuErrchk( cudaDeviceSynchronize() );

//  gpuErrchk( cudaMemcpy(buffer, dev_buffer, sizeof(CudaVec)*w*h, cudaMemcpyDeviceToHost));
//  gpuErrchk( cudaFree(dev_buffer));
}

extern "C"
void CudaGetRayTest(int xx, int yy, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right)
{
  printf("CudaGetRayTest: %d %d %d %d\n", xx, yy, w, h);
  camera.Print(); up.Print(); forward.Print(); right.Print();
  printf("\n");
  CudaRay result;
  CudaRay* dev_result;

  cudaMalloc((void**)&dev_result, sizeof(CudaRay));

  debug_GetRay<<<1, 1>>>(xx, yy, w, h, camera, up, forward, right, dev_result);
//  cudaMemcpy(&result, dev_result, sizeof(CudaRay), cudaMemcpyDeviceToHost);
  result.Print();
  ((result.n.Vec3()+CudaVec(1.0, 1.0, 1.0))/2).Print();
  printf("\n");

  fflush(stdout);
}
