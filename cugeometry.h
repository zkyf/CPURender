#ifndef CUGEOMETRY_H
#define CUGEOMETRY_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <cuda.h>
#include <cuda_runtime.h>
#else
#define CUDA_CALLABLE_MEMBER
#include <math.h>
#endif

struct CudaVec;
struct CudaVec4;
struct CudaVertex;
struct CudaGeometry;
struct CudaRay;

struct CudaVec
{
  double x, y, z;

  CUDA_CALLABLE_MEMBER CudaVec(double _x=0.0, double _y=0.0, double _z=0.0) : x(_x), y(_y), z(_z) {}

#ifdef __CUDACC__
  CUDA_CALLABLE_MEMBER double Dot(const CudaVec& b) { return x*b.x+y*b.y+z*b.z; }
  CUDA_CALLABLE_MEMBER CudaVec Cross(const CudaVec& b) const { return CudaVec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
  CUDA_CALLABLE_MEMBER double Length() const { return sqrt(x*x+y*y+z*z); }
  CUDA_CALLABLE_MEMBER CudaVec Normalized() const { return CudaVec(x/Length(), y/Length(), z/Length()); }
#endif
};

struct CudaVec4
{
  double x, y, z, w;

  CUDA_CALLABLE_MEMBER CudaVec4(double _x=0.0, double _y=0.0, double _z=0.0, double _w=0.0) : x(_x), y(_y), z(_z), w(_w) {}
  CUDA_CALLABLE_MEMBER CudaVec4(CudaVec v3, double _w=0.0) : x(v3.x), y(v3.y), z(v3.z), w(_w) {}
#ifdef __CUDACC__
  CUDA_CALLABLE_MEMBER double Dot(const CudaVec4& b) { return x*b.x+y*b.y+z*b.z+w*b.w; }
  CUDA_CALLABLE_MEMBER double Length() const { return sqrt(x*x+y*y+z*z+w*w); }
  CUDA_CALLABLE_MEMBER CudaVec4 Normalized() const { return CudaVec4(x/Length(), y/Length(), z/Length(), w/Length()); }
  CUDA_CALLABLE_MEMBER CudaVec Vec3() const { return CudaVec(x, y, z); }
#endif
};

#ifdef __CUDACC__
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
#endif

struct CudaVertex
{
  CudaVec4 p;
  CudaVec4 n;
  bool valid = true;
  int geo;

  CUDA_CALLABLE_MEMBER CudaVertex(bool v=true) : valid(v) {}
#ifdef __CUDACC__
  CUDA_CALLABLE_MEMBER static CudaVertex Intersect(CudaVertex a, CudaVertex b, double r)
  {
    CudaVertex result;
    result.n = a.n*r+b.n*(1-r);
    result.p = a.p*r+b.p*(1-r);

    result.valid = a.valid&&b.valid;
    result.geo = a.geo;
  }
#endif
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
#ifdef __CUDACC__
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
#endif
};

struct CudaRay
{
  CudaVec4 o;
  CudaVec4 n;
#ifdef __CUDACC__
  CUDA_CALLABLE_MEMBER CudaVertex IntersectGeo(const CudaGeometry& geo) const
  {
    CudaVec4 pi = geo.Plane();
    double pn = pi.Dot(n);
    if(fabs(pn)<1e-3)
    {
      return CudaVertex(false);
    }
    double r = -pi.Dot(o)/pn;
    if(r<0) return CudaVertex(false);
    CudaVec4 p=r*n+o; // intersection point with the plane

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
    double rb = (ca-cd).Length()/(cb-ce).Length();
    if((ca-cd).Dot(cb-ce)>0) rb=-rb;
    rb = rb/(1+rb);
    double ra = 1-rb;
    CudaVec4 f = rb*geo.vecs[2].p+ra*geo.vecs[1].p;
    double rc = 1-cp.Length()/(f-geo.vecs[0].p).Length();
    if(cp.Dot(f-geo.vecs[0].p)<0) rc=-1;

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
#endif
};

#endif // CUGEOMETRY_H
