#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <ctime>
using namespace std;
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

__device__ int CudaSudoRandomInt(int& seed)
{
  seed++;
//  seed = (seed * 32983 + 92153) % 19483;
//  int result = seed % 19483;
  return seed;
}

__device__ double CudaSudoRandom(int & seed)
{
  int t = CudaSudoRandomInt(seed);
  return (double)(t % 19483) / 19483;
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
  bool selected = false;
  CudaVertex vecs[3];
  CudaVec diffuse = CudaVec(1.0, 1.0, 1.0);
  CudaVec specular = CudaVec();
  CudaVec emission = CudaVec();
  double reflectr = 0.0;
  double refractr = 0.0;
  __device__ CudaVertex Sample(int& seed, double* randNum, int randN) const
  {
    // to do
    CudaVertex result = vecs[0];
    for(int i=1; i<3; i++)
    {
      result = CudaVertex::Intersect(result, vecs[i], randNum[CudaSudoRandomInt(seed)%randN]);
    }
    return result;
  }
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
      if(rb<0)
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
//    if((f-c.p).Length()<1e-3)
//    {
//      result.valid=false;
//      return;
//    }
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

struct CudaKdTree
{
  struct Node
  {
    char axis;

    float xmin = 1e20;
    float ymin = 1e20;
    float zmin = 1e20;

    float xmax = -1e20;
    float ymax = -1e20;
    float zmax = -1e20;

    float value;
    int depth;
    Node* left;
    Node* right;
    Node* parent;
    int n;
    int* gl;
    CudaGeometry* geolist;

    __host__ __device__ Node(Node* p = nullptr) : parent(p) {}
    __host__ __device__ ~Node()
    {
      if(left)
      {
        left->~Node();
      }
      if(right)
      {
        right->~Node();
      }
    }

    __host__ __device__ bool Inside(CudaVec p)
    {
      if(p.x>=xmin && p.x<=xmax && p.y>=ymin && p.y<=ymax && p.z>=zmin && p.z<=zmax)
      {
        return true;
      }
      else
      {
        return false;
      }
    }

    __host__ __device__ void Split()
    {
      if(depth==0)
      {
        for(int gid=0; gid<n; gid++)
        {
          int geo = gl[gid];
          for(int vid=0; vid<3; vid++)
          {
            if(xmin>geolist[geo].vecs[vid].p.x) xmin = geolist[geo].vecs[vid].p.x;
            if(ymin>geolist[geo].vecs[vid].p.y) ymin = geolist[geo].vecs[vid].p.y;
            if(zmin>geolist[geo].vecs[vid].p.z) zmin = geolist[geo].vecs[vid].p.z;

            if(xmax<geolist[geo].vecs[vid].p.x) xmax = geolist[geo].vecs[vid].p.x;
            if(ymax<geolist[geo].vecs[vid].p.y) ymax = geolist[geo].vecs[vid].p.y;
            if(zmax<geolist[geo].vecs[vid].p.z) zmax = geolist[geo].vecs[vid].p.z;
          }
        }
      }

      if(n<=1) { return; }

      float xd = 0.0, xmean = 0.0;
      float yd = 0.0, ymean = 0.0;
      float zd = 0.0, zmean = 0.0;
      int vcount = 0;

      for(int gid=0; gid<n; gid++)
      {
        int geo = gl[gid];
        for(int vid=0; vid<3; vid++)
        {
          if(!Inside(geolist[geo].vecs[vid].p.Vec3())) continue;
          vcount++;
          xmean += geolist[geo].vecs[vid].p.x;
          ymean += geolist[geo].vecs[vid].p.y;
          zmean += geolist[geo].vecs[vid].p.z;
        }
      }

      for(int gid=0; gid<n; gid++)
      {
        int geo = gl[gid];
        for(int vid=0; vid<3; vid++)
        {
          if(!Inside(geolist[geo].vecs[vid].p.Vec3())) continue;
          xd += pow(geolist[geo].vecs[vid].p.x-xmean, 2);
          yd += pow(geolist[geo].vecs[vid].p.y-ymean, 2);
          zd += pow(geolist[geo].vecs[vid].p.z-zmean, 2);
        }
      }

      xd/=vcount; yd/=vcount; zd/=vcount;

    }

  }; // end of CudaKdTree::Node
}; // End of CudaKdTree

#define SampleNum (32)
#define RayTraceLength (5)

static CudaGeometry* dev_geos = nullptr;
static int n = 0;
static FILE* cudaOutput;
static double* hits = nullptr;
static double* dev_hits = nullptr;
static double* dev_randNum = nullptr;
static int randN = 0;
static int* lightGeoList = nullptr;
static int* dev_lightGeoList = nullptr;
static int ln = 0;

__global__ void IntersectGeo(CudaGeometry* geolist, int n, CudaRay ray, double* dev_hits)
{
//  printf("\n");
//  printf("CUDA block (%d, %d, %d) thread (%d %d %d) : ray\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
  if(blockIdx.x<n)
  {
    CudaVertex hp;
    ray.IntersectGeo(geolist[blockIdx.x], hp);
//    ray.Print();
//    printf("\n");
    if(hp.valid)
    {
//      printf("CUDA # %d : hit geo with index # %d\n", blockIdx.x, geolist[blockIdx.x].index);
      geolist[hp.geo].selected = !geolist[hp.geo].selected;
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

__device__ void CudaMonteCarloSample(CudaGeometry* geolist, int n, int* lightGeoList, int ln, CudaVertex o, CudaRay i, double* randNum, int randN, CudaVec& result, int index, bool debuginfo = false)
{
  if(!o.valid) return;
  o.n.w=0;
  o.n = o.n.Normalized();


  CudaVec currentColor=geolist[o.geo].diffuse;

  bool haslight=geolist[o.geo].emission.Length()>0.1;
  CudaRay currentIn = i;

  if(haslight)
  {
    result = geolist[o.geo].emission;
    return;
  }

  for(int level=0; level<RayTraceLength; level++)
  {
    if(o.n.Dot(currentIn.n)>0) o.n = -1.0*o.n;
    if(debuginfo)
    {
      printf("======================================\n");
      printf("Current Geo # %d: ", o.geo);
      printf("reflect %.6lf refract %.6lf\n", geolist[o.geo].reflectr, geolist[o.geo].refractr);
      printf("Current Level: %d, index = %d\n", level, index);
      printf("CurrentO = "); o.p.Print(); printf("\n");
      printf("CurrentColor = "); currentColor.Print();
      printf("\n");
    }

    CudaRay ray;
    ray.o=o.p;

    if(geolist[o.geo].reflectr>1e-3)
    {
      ray.n = (currentIn.n-2*o.n*(currentIn.n.Dot(o.n))).Normalized();
    }
    else
    {
      float sin2theta=randNum[CudaSudoRandomInt(index)%randN]; // sin^2(theta)
      float sintheta=sqrt(sin2theta);
      float phi = randNum[CudaSudoRandomInt(index)%randN]*2*M_PI;
      if(debuginfo)
      {
        printf("sin2theta = %.6lf, phi = %.6lf, index = %d\n", sin2theta, phi, index);
      }
      CudaVec w = o.n.Vec3();
      CudaVec u = (fabs(w.x)>0.1?CudaVec(0, 1):CudaVec(1)).Cross(w);
      CudaVec v = w.Cross(u);
      ray.n = CudaVec4(sintheta*cos(phi)*u+v*sintheta*sin(phi)+w*sqrt(1-sin2theta), 0);
      if(debuginfo)
      {
        printf("w = "); w.Print(); printf(", u = "); u.Print(); printf(", v = "); v.Print(); printf("\n");
        printf("generated ray: "); ray.Print(); printf("\n");
      }
    }

    double mind=1e20;
    CudaVertex minp(false);

    for(int i=0; i<n; i++)
    {
      CudaVertex hp;
      ray.IntersectGeo(geolist[i], hp);
      if(hp.valid)
      {
        if((hp.p-o.p).Length()>1e-3 && (hp.p-o.p).Length()<mind)
        {
          mind=(hp.p-o.p).Length();
          minp=hp;
        }
      }
    }

    if(minp.valid)
    {
      if(debuginfo)
      {
        printf("Ray hit geo # %d @", minp.geo);
        minp.p.Print();
        printf("\n");
      }

      if(geolist[minp.geo].emission.Length()>0.1)
      {
        float r = 1.0/(minp.p-o.p).Length();
        r = r*r;
        if(r>1/fabs(ray.n.Dot(o.n))) r=1/fabs(ray.n.Dot(o.n));
        currentColor = currentColor * geolist[minp.geo].emission * fabs(ray.n.Dot(o.n)) * r * 3;
        if(debuginfo)
        {
          printf("got light color: "); geolist[minp.geo].emission.Print(); printf("\n");
          printf("current color = "); currentColor.Print(); printf("\n");
        }
        haslight=true;
      }
      else
      {
        float r = 1.0/(minp.p-o.p).Length();
        r = r*r;
        if(r>1) r=1;
        currentColor = currentColor * geolist[minp.geo].diffuse * fabs(ray.n.Dot(o.n)) * r * 0.9;
        if(debuginfo)
        {
          printf("got diffuse color: "); geolist[minp.geo].diffuse.Print(); printf("\n");
          printf("current color = "); currentColor.Print(); printf("\n");
        }
      }
    }
    else
    {
      if(debuginfo)
      {
        printf("ray hit nothing\n");
        printf("======================================\n\n");
      }
      break;
    }

    if(debuginfo)
    {
      printf("======================================\n\n");
    }

    o = minp;
    currentIn = ray;
  }

  if(haslight)
  {
    if(debuginfo)
    {
      printf("FINAL: haslight "); currentColor.Print(); printf("\n");
    }
    result = currentColor;
    return;
  }
  else
  {
//    result = currentColor;
    if(debuginfo)
    {
      printf("FINAL: NOT haslight "); currentColor.Print(); printf("\n");
      printf("# %d lightGeo in total\n", ln);
    }
    result = CudaVec(0, 0, 0);
    if(o.valid)
    {
      if(geolist[o.geo].reflectr>1e-3)
      {
        return;
      }

      int hitcount=0;
      CudaVec lightColor;
      for(int ii=0; ii<ln; ii++)
      {
        if(debuginfo)
        {
          printf("\n\ntest lightGeo # %d\n", lightGeoList[ii]);
        }
//        for(int j=0; j<; j++)
        {
          CudaVertex v = geolist[lightGeoList[ii]].Sample(index, randNum, randN);
          if(debuginfo)
          {
//            printf("  sample vertex # %d\n", j);
            printf("  d = %.6lf, ", (v.p-o.p).Length()); v.p.Print(); printf("\n");
          }
          CudaRay ray; ray.o = o.p; ray.n = CudaVec4((v.p.Vec3()-o.p.Vec3()).Normalized(), 0.0);
          bool visible = true;
          int lighthit = lightGeoList[ii];
          double d = (v.p-o.p).Length();
          for(int k=0; k<n; k++)
          {
            if(k==lightGeoList[ii]) continue;
            CudaVertex ir;
            ray.IntersectGeo(geolist[k], ir);
            if(ir.valid && (ir.p-o.p).Length()<d && (ir.p-o.p).Length()>1e-3)
            {
              v = ir;
              d = (ir.p-o.p).Length();
              if(debuginfo)
              {
                printf("  "); ray.Print();
                printf("  geo # %d got hit @ ", k); ir.p.Print(); printf(" d = %lf emission = %.6lf\n\n", d, geolist[k].emission.Length());
              }
              if(geolist[k].emission.Length()>0.1)
              {
                lighthit = k;
                visible = true;
              }
              else
              {
                visible = false;
              }
            }
          }
          if(visible)
          {
            hitcount++;
            float r = 1.0/d*d;
            if(r>1/fabs(ray.n.Dot(o.n))) r=1/fabs(ray.n.Dot(o.n));
            lightColor = lightColor + geolist[lighthit].emission * fabs(ray.n.Dot(o.n))*r * 3;
            if(debuginfo)
            {
              printf("lightgeo # %d visible", lighthit); (geolist[lighthit].emission * fabs(ray.n.Dot(o.n))).Print(); printf("\n");
              printf("now light color = "); lightColor.Print(); printf("\n");
            }
          }
        }
      }
      if(hitcount>0)
      {
        result = currentColor * lightColor / hitcount;
        if(debuginfo)
        {
          printf("\n\nFinal result color = "); result.Print(); printf("\n");
        }
      }
    }
    return;
  }
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

__global__ void CudaMonteCarloRender(CudaGeometry* geolist, int n, int* lightGeoList, int ln, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer, double* randNum, int randN, bool debuginfo = false)
{
  __shared__ CudaVec tResults[SampleNum];
  if(n==0) return;
  int xx=blockIdx.x;
  int yy=blockIdx.y;
  int tid=threadIdx.x;
  tResults[tid] = CudaVec(0, 0, 0);

  if(xx<0 || xx>=w || yy<0 || yy>=h) return;
  int index = xx+yy*w;
  int seed = index*blockDim.x+threadIdx.x;
  CudaRay ray = GetRay(xx, yy, w, h, camera, up, forward, right);
//  buffer[index] = CudaVec(0, 0, 0);

//  for(int sp=0; sp<SampleNum; sp++)
  {
    double mind=1e20;
    CudaVertex minp(false);

//    buffer[index] = buffer[index]+(ray.n.Vec3()+CudaVec(1.0, 1.0, 1.0))/2;

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
      CudaMonteCarloSample(geolist, n, lightGeoList, ln, minp, ray, randNum, randN, result, seed * 17, debuginfo);
      tResults[tid] = result;
    }
  }

  if(tid==0)
  {
    int count=1;
    for(int i=1; i<blockDim.x; i++)
    {
      if(tResults[tid].Length()>1e-3)
      tResults[tid] = tResults[tid]+tResults[i];
      count++;
    }
    buffer[index] = tResults[tid]/count;
  }
  __syncthreads();

  return;
}
__global__ void CudaDivide(int w, int h, CudaVec* buffer)
{
  int xx=blockIdx.x;
  int yy=blockIdx.y;
  if(xx<0 || xx>=w || yy<0 || yy>=h) return;
  int index = xx+yy*gridDim.x;
  buffer[index] = buffer[index]/4;
}

__global__ void debug_MonteCarloRender(CudaGeometry* geolist, int n, int* lightGeoList, int ln, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer, double* randNum, int randN, bool debuginfo = false, int _xx=-1, int _yy=1)
{
  int xx = _xx;
  int yy = _yy;
  int index = xx+yy*gridDim.x;
  if(xx<0 || xx>=w || yy<0 || yy>=h) return;
  CudaRay ray = GetRay(xx, yy, w, h, camera, up, forward, right);
  buffer[index] = CudaVec(0, 0, 0);

  printf("debug_MonteCarloRender @ %d %d\n", xx, yy);

  for(int sp=0; sp<1; sp++)
  {
    double mind=1e20;
    CudaVertex minp(false);

//    buffer[index] = buffer[index]+(ray.n.Vec3()+CudaVec(1.0, 1.0, 1.0))/2;

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
      CudaMonteCarloSample(geolist, n, lightGeoList, ln, minp, ray, randNum, randN, result, index, debuginfo);
      buffer[index] = buffer[index] + result;
    }
  }

//  buffer[index] = buffer[index]/SampleNum;

  return;
}

extern "C" void CudaMonteCarloSampleTest(int xx, int yy, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right)
{
  float elapsed=0;
  cudaEvent_t start, stop;
  CudaVec* dev_buffer;

  gpuErrchk( cudaMalloc((void**)&dev_buffer, sizeof(CudaVec)*(w*h+10)));

  printf("start cuda render\n");
  fflush(stdout);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
//  for(int i=0; i<4; i++)
//  {
    debug_MonteCarloRender<<<1, 1>>>(dev_geos, n, dev_lightGeoList, ln, w, h, camera, up, forward, right, dev_buffer, dev_randNum, randN, true, xx, yy);
//  }
//  CudaDivide<<<dim3(w, h), 1>>>(w, h, dev_buffer);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("The elapsed time in gpu was %.2f ms", elapsed);
  fflush(stdout);
  printf("end cuda render\n");
  fflush(stdout);

//  gpuErrchk( cudaPeekAtLastError() );
//  gpuErrchk( cudaDeviceSynchronize() );

  gpuErrchk( cudaFree(dev_buffer));
}

extern "C"
void CudaInit(CudaGeometry* geos, int _n, int* lightList, int _ln, double* h, double* randNum, int _randN)
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
    gpuErrchk(cudaFree(dev_randNum));
    gpuErrchk(cudaFree(dev_lightGeoList));
  }

  n = _n;
  randN = _randN;
  ln = _ln;

  printf("CUDA: sizeof(CudaGeometry)=%d, sizeof(CudaVec)=%d, n=%d\n", sizeof(CudaGeometry), sizeof(CudaVec), n);

  gpuErrchk( cudaMalloc((void**)&dev_geos, sizeof(CudaGeometry)*n));
  gpuErrchk( cudaMalloc((void**)&dev_hits, sizeof(double)*n));
  gpuErrchk( cudaMalloc((void**)&dev_lightGeoList, sizeof(int)*ln));
  gpuErrchk( cudaMalloc((void**)&dev_randNum, sizeof(double)*randN));
  gpuErrchk( cudaMemcpy(dev_geos, geos, sizeof(CudaGeometry)*n, cudaMemcpyHostToDevice));
  gpuErrchk( cudaMemcpy(dev_randNum, randNum, sizeof(double)*randN, cudaMemcpyHostToDevice));
  gpuErrchk( cudaMemcpy(dev_lightGeoList, lightList, sizeof(int)*ln, cudaMemcpyHostToDevice));
  gpuErrchk( cudaPeekAtLastError() );
  fflush(stdout);
}

extern "C"
void CudaEnd()
{
  if(dev_geos)
  {
    gpuErrchk(cudaFree(dev_geos));
    gpuErrchk(cudaFree(dev_hits));
    gpuErrchk(cudaFree(dev_randNum));
    gpuErrchk(cudaFree(dev_lightGeoList));
  }
}

extern "C" void CudaRender(int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer)
{
  float elapsed=0;
  cudaEvent_t start, stop;
  CudaVec* dev_buffer;
  memset(buffer, 0, sizeof(CudaVec)*w*h);
  gpuErrchk( cudaMalloc((void**)&dev_buffer, sizeof(CudaVec)*(w*h+10)));

  printf("start cuda render\n");
  printf("buffer.size=%d %d\n", sizeof(buffer), sizeof(buffer)/sizeof(CudaVec));
  fflush(stdout);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
//  for(int i=0; i<4; i++)
//  {
    CudaMonteCarloRender<<<dim3(w, h), SampleNum>>>(dev_geos, n, dev_lightGeoList, ln, w, h, camera, up, forward, right, dev_buffer, dev_randNum, randN);
//  }
//  CudaDivide<<<dim3(w, h), 1>>>(w, h, dev_buffer);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("The elapsed time in gpu was %.2f ms", elapsed);
  fflush(stdout);
  printf("end cuda render\n");
  fflush(stdout);

//  gpuErrchk( cudaPeekAtLastError() );
//  gpuErrchk( cudaDeviceSynchronize() );

  clock_t begin = clock();
  gpuErrchk( cudaMemcpy(buffer, dev_buffer, sizeof(CudaVec)*(w*h+10), cudaMemcpyDeviceToHost));
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  printf("cudaMemcpy costs %.6lf secs.\n", elapsed_secs);
  fflush(stdout);
  gpuErrchk( cudaFree(dev_buffer));
}

extern "C" void CudaGetRayTest(int xx, int yy, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right)
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

extern "C" void CudaIntersect(CudaRay ray)
{
  printf("CudaIntersect n=%d\n", n);
  ray.Print();
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
