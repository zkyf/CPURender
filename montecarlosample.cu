#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <ctime>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

using namespace std;
//#include "cugeometry.h"

#define M_PI (3.1415926)
#define EPS (1e-3)

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

template<class T> class CudaLinkedList
{
private:
  struct Container
  {
    T data;
    Container* next;
    Container* prev;

    __host__ __device__ Container() : next(nullptr), prev(nullptr) {}
    __host__ __device__ ~Container() {}
  };
  Container* head;
  Container* tail;
  int count;

public:
  typedef Container* iterator;

  __host__ __device__ CudaLinkedList() : head(nullptr), tail(nullptr), count(0) {}

  __host__ __device__ void push_back(const T& t)
  {
    Container* nl = new Container();
    nl->data = t;
    nl->next = nullptr;
    if(tail==nullptr)
    {
      tail=nl;
      head=nl;
      nl->prev=nullptr;
    }
    else
    {
      nl->prev = tail;
      tail->next = nl;
    }
    tail = nl;
    count++;
  }

  __host__ __device__ void push_front(const T& t)
  {
    Container* nl = new Container();
    nl->data = t;
    nl->prev = nullptr;
    if(head==nullptr)
    {
      tail=nl;
      head=nl;
      nl->next=nullptr;
    }
    else
    {
      nl->next = head;
      head->prev = nl;
    }
    head = nl;
    count++;
  }

  __host__ __device__ T pop_back()
  {
    Container* todelete = tail;
    T data = todelete->data;
    tail = tail->prev;
    delete todelete;
    count--;
    return data;
  }

  __host__ __device__ T pop_head()
  {
    Container* todelete = head;
    T data = todelete->data;
    head = head->next;
    delete todelete;
    count--;
    return data;
  }

  __host__ __device__ int size() { return count; }

  __host__ __device__ void clear() { while(count>0) pop_head(); }

  __host__ __device__ iterator begin() { return head; }

  __host__ __device__ iterator end() { return nullptr; }
};

template<class T> class CudaVector
{
private:
#define _CudaVector_DefaultSize (20)

  T* head;
  int count;
  int s;

  __host__ __device__ void MoveToNewArray(int ns)
  {
    T* nhead = new T[ns];
    if(head!=nullptr)
    {
      for(int i=0; i<s; i++)
      {
        nhead[i] = head[i];
      }
      delete[] head;
    }
    head = nhead;
    s = ns;
  }

  __host__ __device__ void MoveRange(T* a, int n, int delta)
  {
    if(delta==0) return;
    if(delta>0)
    {
      // for insertion
      for(int i=n-1; i>=0; i++)
      {
        a[i+delta]=a[i];
      }
    }
    else
    {
      // for removal
      for(int i=0; i<n; i++)
      {
        a[i+delta]=a[i];
      }
    }
  }

public:

  typedef T* iterator;

  __host__ __device__ CudaVector() : head(nullptr)
  {
    count=0;
    MoveToNewArray(_CudaVector_DefaultSize);
  }

  __host__ __device__ CudaVector(const T& t, int _s) : head(nullptr)
  {
    count=_s;
    MoveToNewArray(_s);
    for(int i=0; i<_s; i++)
    {
      head[i] = t;
    }
  }

  __host__ __device__ CudaVector(CudaVector& b)
  {
    count = b.count;
    s = b.s;
    if(head)
    {
      delete[] head;
    }
    MoveToNewArray(s);
    for(int i=0; i<s; i++)
    {
      head[i] = b.head[i];
    }
  }

  __host__ __device__ ~CudaVector()
  {
    if(head) delete[] head;
  }

  __host__ __device__ iterator begin()
  {
    return head;
  }

  __host__ __device__ iterator end()
  {
    return head+s;
  }

  __host__ __device__ T& first()
  {
    return head[0];
  }

  __host__ __device__ T& last()
  {
    return head[count-1];
  }

  __host__ __device__ void push_back(const T& t)
  {
    if(count==s)
    {
      MoveToNewArray(s+_CudaVector_DefaultSize);
    }
    head[count]=t;
    count++;
  }

  __host__ __device__ void remove(int index)
  {
    if(index!=count-1)
    {
      MoveRange(head+index+1, count-index-1, -1);
    }
    count--;
  }

  __host__ __device__ void clear()
  {
    count=0;
  }

  __host__ __device__ int size()
  {
    return count;
  }

  __host__ __device__ T& at(const int& index)
  {
    return head[index];
  }

  __host__ __device__ T& operator[](const int& index)
  {
    return head[index];
  }

  __host__ __device__ void sort()
  {
    for(int i=0; i<count; i++)
    {
      for(int j=i+1; j<count; j++)
      {
        if(head[j]<head[i])
        {
          T temp=head[j];
          head[j]=head[i];
          head[i]=temp;
        }
      }
    }
  }
};

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
  double ni = 1.0;
  __device__ CudaVertex Sample(curandState_t* randstate, double* randNum, int randN) const
  {
    // to do
    CudaVertex result = vecs[0];
    for(int i=1; i<3; i++)
    {
      result = CudaVertex::Intersect(result, vecs[i], curand_uniform(randstate));
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
  float d = 1.0;

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

    if(fabs(pn)<EPS)
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

    if((c.p-p).Vec3().Length()<EPS)
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
    if((cb-ce).Length()>EPS)
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
//    if((f-c.p).Length()<EPS)
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

  __host__ __device__ void Print() const { printf("CudaRay: o=(%lf, %lf, %lf, %lf) , n=(%lf, %lf, %lf, %lf)\n", o.x, o.y, o.z, o.w, n.x, n.y, n.z, n.w); }
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
    CudaVector<int> gl;
    CudaGeometry* geolist;

    __host__ __device__ Node(Node* p = nullptr) : parent(p), left(nullptr), right(nullptr), depth(0) {}
    __host__ __device__ ~Node()
    {
      if(left)
      {
        left->~Node();
        delete left;
      }
      if(right)
      {
        right->~Node();
        delete right;
      }
    }

    __host__ __device__ bool Inside(CudaVec p)
    {
      if(p.x>=xmin-EPS && p.x<=xmax+EPS && p.y>=ymin-EPS && p.y<=ymax+EPS && p.z>=zmin-EPS && p.z<=zmax+EPS)
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

    } // end of CudaKdTree::Node::Split()

    __host__ __device__ CudaVertex HitChild(CudaRay ray, bool debuginfo = false)
    {
      if(left!=nullptr && right!=nullptr)
      {
        CudaVertex lr = left->Hit(ray, debuginfo);
        CudaVertex rr = right->Hit(ray, debuginfo);
        if((lr.valid && !rr.valid) || (lr.valid && rr.valid && (lr.p-ray.o).Length()<(rr.p-ray.o).Length())) return lr;
        else return rr;
      }
      else
      {
        float min=1e20;
        int ming=-1;
        CudaVertex minp(false);
        for(int gid=0; gid<gl.size(); gid++)
        {
          CudaGeometry& geo = geolist[gl[gid]];
          CudaVertex hp;
          ray.IntersectGeo(geo, hp);

          if(hp.valid)
          {
            float d=(hp.p-ray.o).Length();
            if(d<min && d>EPS)
            {
              min=d;
              ming=gl[gid];
              minp=hp;
            }
          }
        }

        return minp;
      }
    }

    __host__ __device__ bool Hit(CudaRay ray, bool debuginfo = false)
    {
      if(debuginfo)
      {
        printf("CudaKdTree::Node::Hit @ depth=%d this=%x\n", depth, this);
        printf("xmin=%.6lf, xmax=%.6lf\n", xmin, xmax);
        printf("ymin=%.6lf, ymax=%.6lf\n", ymin, ymax);
        printf("zmin=%.6lf, zmax=%.6lf\n", zmin, zmax);
        ray.Print();
      }
      ray.o.w = 1.0;
      ray.n.w = 0.0;
      if(Inside(ray.o.Vec3()))
      {
        if(debuginfo)
        {
          printf("  ray starts in side\n");
        }
//        CudaVertex result = HitChild(ray);
        return true;
      }
      else
      {
        if(ray.o.x<xmin-EPS && ray.n.x<0 ||
           ray.o.x>xmax+EPS && ray.n.x>0 ||

           ray.o.y<ymin-EPS && ray.n.y<0 ||
           ray.o.y>ymax+EPS && ray.n.y>0 ||

           ray.o.z<zmin-EPS && ray.n.z<0 ||
           ray.o.z>zmax+EPS && ray.n.z>0)
        {
          if(debuginfo)
          {
            printf("  ray starts outside and go further\n");
          }
//          return CudaVertex(false);
          return false;
        }

        if(fabs(ray.n.x)>=fabs(ray.n.y) && fabs(ray.n.x)>fabs(ray.n.z))
        {
          CudaVec4 h_min = (xmin-ray.o.x)/ray.n.x*ray.n+ray.o;
          CudaVec4 h_max = (xmax-ray.o.x)/ray.n.x*ray.n+ray.o;

          if(debuginfo)
          {
            printf("test x\n");
            printf("  h_min="); h_min.Print(); printf(", h_max="); h_max.Print();
            printf("\n");
          }

          if( ((ymin<=h_max.y+EPS && ymax>=h_min.y-EPS) || (ymin<=h_min.y+EPS && ymax>=h_max.y-EPS) ) &&
              ((zmin<=h_max.z+EPS && zmax>=h_min.z-EPS) || (zmin<=h_min.z+EPS && zmax>=h_max.z-EPS) ))
          {
            // intersect!
            if(debuginfo)
            {
              printf("  hit!\n");
            }
//            CudaVertex ir = HitChild(ray, debuginfo);
            return true;
          }
          else
          {
//            return CudaVertex(false);
            return false;
          }
        }
        else if(fabs(ray.n.y)>fabs(ray.n.x) && fabs(ray.n.y)>fabs(ray.n.z))
        {
          CudaVec4 h_min = (ymin-ray.o.y)/ray.n.y*ray.n+ray.o;
          CudaVec4 h_max = (ymax-ray.o.y)/ray.n.y*ray.n+ray.o;

          if(debuginfo)
          {
            printf("test y\n");
            printf("  h_min="); h_min.Print(); printf(", h_max="); h_max.Print();
            printf("\n");
          }

          if( ((xmin<=h_max.x+EPS && xmax>=h_min.x-EPS) || (xmin<=h_min.x+EPS && xmax>=h_max.x-EPS) ) &&
              ((zmin<=h_max.z+EPS && zmax>=h_min.z-EPS) || (zmin<=h_min.z+EPS && zmax>=h_max.z-EPS) ))
          {
            // intersect!
            if(debuginfo)
            {
              printf("  hit!\n");
            }
//            CudaVertex ir = HitChild(ray, debuginfo);
            return true;
          }
          else
          {
//            return CudaVertex(false);
            return false;
          }
        }
        else
        {
          CudaVec4 h_min = (zmin-ray.o.z)/ray.n.z*ray.n+ray.o;
          CudaVec4 h_max = (zmax-ray.o.z)/ray.n.z*ray.n+ray.o;

          if(debuginfo)
          {
            printf("test z\n");
            printf("  h_min="); h_min.Print(); printf(", h_max="); h_max.Print();
            printf("\n");
          }

          if( ((xmin<=h_max.x+EPS && xmax>=h_min.x-EPS) || (xmin<=h_min.x+EPS && xmax>=h_max.x-EPS) ) &&
              ((ymin<=h_max.y+EPS && ymax>=h_min.y-EPS) || (ymin<=h_min.y+EPS && ymax>=h_max.y-EPS) ))
          {
            // intersect!
            if(debuginfo)
            {
              printf("  hit!\n");
            }
//            CudaVertex ir = HitChild(ray, debuginfo);
            return true;
          }
          else
          {
            return false;
          }
        }
      }
    } // end of CudaKdTree::Node::Hit()

    __host__ __device__ void Print()
    {
      printf("CudaKdTree::Node depth=%d, axis=%d, value=%lf\n", depth, axis, value);
      printf("gl.size()=%d\n", gl.size());
      for(int i=0; i<gl.size(); i++)
      {
        printf("gl # %d : %d\n", i, gl[i]);
      }
      if(left)
      {
        printf("\n\nleft chiild: ");
        left->Print();
      }
      if(right)
      {
        printf("\n\nright chiild: ");
        right->Print();
      }
    }
  }; // end of CudaKdTree::Node

  Node* root;
  CudaGeometry* geolist;
  int n;

  __host__ __device__ CudaKdTree() : root(nullptr), geolist(nullptr), n(-1) {}
  __host__ __device__ CudaKdTree(CudaGeometry* gl, int _n) : geolist(gl), n(_n)
  {
    root = new Node(nullptr);
    root->depth=1;
    root->geolist=gl;
    for(int i=0; i<n; i++)
    {
      root->gl.push_back(i);
    }
    root->Split();
  }
  __host__ __device__ ~CudaKdTree()
  {
    if(root)
    {
      root->~Node();
    }
  }

  __host__ __device__ void SetTree(CudaGeometry* gl, int _n)
  {
    printf("CudaKdTree::SetTree()\n");
    geolist = gl;
    n=_n;
    if(root)
    {
      delete root;
      root->~Node();
    }

    root = new Node(nullptr);
    root->depth=0;
    root->geolist=gl;
    for(int i=0; i<n; i++)
    {
      root->gl.push_back(i);
    }

    CudaVector<Node*> queue;
    queue.push_back(root);
    while(queue.size()>0)
    {
      printf("\n==============================\n");
      Node* now = queue[0];
      queue.remove(0);
      printf("this = %x\n", now);

      now->left = nullptr;
      now->right = nullptr;
      printf("CudaKdTree::SetTree depth=%d\n", now->depth);
      printf("parent = %x\n", now->parent);
      if(now->depth==0)
      {
        now->xmin = 1e20; now->xmax = -1e20;
        now->ymin = 1e20; now->ymax = -1e20;
        now->zmin = 1e20; now->zmax = -1e20;
        printf("gl.size=%d\n", now->gl.size());
        for(int gid=0; gid<now->gl.size(); gid++)
        {
          int geo = now->gl[gid];
          for(int vid=0; vid<3; vid++)
          {
//            printf("geo # %d vid # %d : ", gl[gid], vid);
//            geolist[geo].vecs[vid].p.Print();
//            printf("\n");
            if(now->xmin>geolist[geo].vecs[vid].p.x) now->xmin = geolist[geo].vecs[vid].p.x;
            if(now->ymin>geolist[geo].vecs[vid].p.y) now->ymin = geolist[geo].vecs[vid].p.y;
            if(now->zmin>geolist[geo].vecs[vid].p.z) now->zmin = geolist[geo].vecs[vid].p.z;

            if(now->xmax<geolist[geo].vecs[vid].p.x) now->xmax = geolist[geo].vecs[vid].p.x;
            if(now->ymax<geolist[geo].vecs[vid].p.y) now->ymax = geolist[geo].vecs[vid].p.y;
            if(now->zmax<geolist[geo].vecs[vid].p.z) now->zmax = geolist[geo].vecs[vid].p.z;
          }
        }
      }
      printf("xmin=%.6lf, xmax=%.6lf\n", now->xmin, now->xmax);
      printf("ymin=%.6lf, ymax=%.6lf\n", now->ymin, now->ymax);
      printf("zmin=%.6lf, zmax=%.6lf\n", now->zmin, now->zmax);

      if(now->gl.size()<=1) { continue; }
//      if(now->depth>=10) { continue; }

      float xd = 0.0, xmean = 0.0;
      float yd = 0.0, ymean = 0.0;
      float zd = 0.0, zmean = 0.0;
      int vcount = 0;

      for(int gid=0; gid<now->gl.size(); gid++)
      {
        int geo = now->gl[gid];
        for(int vid=0; vid<3; vid++)
        {
          if(!now->Inside(geolist[geo].vecs[vid].p.Vec3())) continue;
          vcount++;
          xmean += geolist[geo].vecs[vid].p.x;
          ymean += geolist[geo].vecs[vid].p.y;
          zmean += geolist[geo].vecs[vid].p.z;
        }
      }
      xmean/=vcount; ymean/=vcount; zmean/=vcount;

      for(int gid=0; gid<now->gl.size(); gid++)
      {
        int geo = now->gl[gid];
        for(int vid=0; vid<3; vid++)
        {
          if(!now->Inside(geolist[geo].vecs[vid].p.Vec3())) continue;
          xd += pow(geolist[geo].vecs[vid].p.x-xmean, 2);
          yd += pow(geolist[geo].vecs[vid].p.y-ymean, 2);
          zd += pow(geolist[geo].vecs[vid].p.z-zmean, 2);
        }
      }

      xd/=vcount; yd/=vcount; zd/=vcount;

      printf("vcount = %d\n", vcount);
      printf("xmean=%.6lf, xd=%.6lf\n", xmean, xd);
      printf("ymean=%.6lf, yd=%.6lf\n", ymean, yd);
      printf("zmean=%.6lf, zd=%.6lf\n", zmean, zd);

      now->axis =0;
      float nd=xd;
      if(yd>nd) { now->axis=1; nd=yd; }
      if(zd>nd) { now->axis=2; nd=zd; }

      bool continueSplit = true;

      CudaVector<double> temp;

      switch(now->axis)
      {
      case 0: // split on x
        printf("split on x\n");
        temp.clear();
        for(int gid=0; gid<now->gl.size(); gid++)
        {
          for(int vid=0; vid<3; vid++)
          {
            if(!now->Inside(geolist[now->gl[gid]].vecs[vid].p.Vec3()))
            {
              continue;
            }
            temp.push_back(geolist[now->gl[gid]].vecs[vid].p.x);
          }
        }

        printf("sorting temp # %d\n", temp.size());
        temp.sort();

        if(fabs(temp.first() - temp.last())<EPS)
        {
          continueSplit = false;
          break;
        }
        else
        {
          now->left  = new Node(now); now->left->geolist  = now->geolist;
          now->right = new Node(now); now->right->geolist = now->geolist;
          printf("continue split left= %x right= %x\n", now->left, now->right);
        }

        for(int i=temp.size()-1; i>0; i--)
        {
          if(fabs(temp[i]-temp[i-1])<EPS)
          {
            temp.remove(i);
          }
        }

        for(int i=0; i<temp.size(); i++)
        {
//          printf("# %d : %.6lf\n", i, temp[i]);
        }

        if(temp.size()%2==1)
        {
          now->value = temp[temp.size()/2];
        }
        else
        {
          now->value = (temp[temp.size()/2] + temp[temp.size()/2-1])/2;
        }

        for(int i=0; i<now->gl.size(); i++)
        {
          CudaGeometry& geo = geolist[now->gl[i]];
          bool leftused = false;
          bool rightused = false;
          for(int vid=0; vid<3; vid++)
          {
            if(geo.vecs[vid].p.x<=now->value && !leftused)
            {
              now->left->gl.push_back(now->gl[i]);
              leftused = true;
            }
            else if(geo.vecs[vid].p.x>=now->value && !rightused)
            {
              now->right->gl.push_back(now->gl[i]);
              rightused = true;
            }
            if(leftused && rightused) break;
          }
        }

        if(continueSplit)
        {
          now->left->xmin=now->xmin;
          now->left->xmax=now->value;
          now->left->ymin=now->ymin;
          now->left->ymax=now->ymax;
          now->left->zmin=now->zmin;
          now->left->zmax=now->zmax;

          now->right->xmin=now->value;
          now->right->xmax=now->xmax;
          now->right->ymin=now->ymin;
          now->right->ymax=now->ymax;
          now->right->zmin=now->zmin;
          now->right->zmax=now->zmax;
        }
        break; // end of split on x

      case 1:
        // split on y
        printf("split on y\n");
        temp.clear();
        for(int gid=0; gid<now->gl.size(); gid++)
        {
          for(int vid=0; vid<3; vid++)
          {
            if(!now->Inside(now->geolist[now->gl[gid]].vecs[vid].p.Vec3()))
            {
              continue;
            }
            temp.push_back(now->geolist[now->gl[gid]].vecs[vid].p.y);
          }
        }

        printf("sorting temp # %d\n", temp.size());
        temp.sort();

        if(fabs(temp.first() - temp.last())<EPS)
        {
          continueSplit = false;
          break;
        }
        else
        {
          now->left  = new Node(now); now->left->geolist  = now->geolist;
          now->right = new Node(now); now->right->geolist = now->geolist;
          printf("continue split left= %x right= %x\n", now->left, now->right);
        }

        for(int i=temp.size()-1; i>0; i--)
        {
          if(fabs(temp[i]-temp[i-1])<EPS)
          {
            temp.remove(i);
          }
        }

        for(int i=0; i<temp.size(); i++)
        {
//          printf("# %d : %.6lf\n", i, temp[i]);
        }

        if(temp.size()%2==1)
        {
          now->value = temp[temp.size()/2];
        }
        else
        {
          now->value = (temp[temp.size()/2] + temp[temp.size()/2-1])/2;
        }

        for(int i=0; i<now->gl.size(); i++)
        {
          CudaGeometry& geo = geolist[now->gl[i]];
          bool leftused = false;
          bool rightused = false;
          for(int vid=0; vid<3; vid++)
          {
            if(geo.vecs[vid].p.y<=now->value && !leftused)
            {
              now->left->gl.push_back(now->gl[i]);
              leftused = true;
            }
            else if(geo.vecs[vid].p.y>=now->value && !rightused)
            {
              now->right->gl.push_back(now->gl[i]);
              rightused = true;
            }
            if(leftused && rightused) break;
          }
        }

        if(continueSplit)
        {
          now->left->ymin=now->ymin;
          now->left->ymax=now->value;
          now->left->xmin=now->xmin;
          now->left->xmax=now->xmax;
          now->left->zmin=now->zmin;
          now->left->zmax=now->zmax;

          now->right->ymin=now->value;
          now->right->ymax=now->ymax;
          now->right->xmin=now->xmin;
          now->right->xmax=now->xmax;
          now->right->zmin=now->zmin;
          now->right->zmax=now->zmax;
        }
        break; // end of split on y

      case 2:
        // split on y
        printf("split on z\n");
        temp.clear();
        for(int gid=0; gid<now->gl.size(); gid++)
        {
          for(int vid=0; vid<3; vid++)
          {
            if(!now->Inside(now->geolist[now->gl[gid]].vecs[vid].p.Vec3()))
            {
              continue;
            }
            temp.push_back(now->geolist[now->gl[gid]].vecs[vid].p.z);
          }
        }

        printf("sorting temp # %d\n", temp.size());
        temp.sort();

        if(fabs(temp.first() - temp.last())<EPS)
        {
          continueSplit = false;
          break;
        }
        else
        {
          now->left  = new Node(now); now->left->geolist  = now->geolist;
          now->right = new Node(now); now->right->geolist = now->geolist;
          printf("continue split left= %x right= %x\n", now->left, now->right);
        }

        for(int i=temp.size()-1; i>0; i--)
        {
          if(fabs(temp[i]-temp[i-1])<EPS)
          {
            temp.remove(i);
          }
        }

        for(int i=0; i<temp.size(); i++)
        {
//          printf("# %d : %.6lf\n", i, temp[i]);
        }

        if(temp.size()%2==1)
        {
          now->value = temp[temp.size()/2];
        }
        else
        {
          now->value = (temp[temp.size()/2] + temp[temp.size()/2-1])/2;
        }

        for(int i=0; i<now->gl.size(); i++)
        {
          CudaGeometry& geo = now->geolist[now->gl[i]];
          bool leftused = false;
          bool rightused = false;
          for(int vid=0; vid<3; vid++)
          {
            if(geo.vecs[vid].p.z<=now->value && !leftused)
            {
              now->left->gl.push_back(now->gl[i]);
              leftused = true;
            }
            else if(geo.vecs[vid].p.z>=now->value && !rightused)
            {
              now->right->gl.push_back(now->gl[i]);
              rightused = true;
            }
            if(leftused && rightused) break;
          }
        }

        if(continueSplit)
        {
          now->left->zmin=now->zmin;
          now->left->zmax=now->value;
          now->left->xmin=now->xmin;
          now->left->xmax=now->xmax;
          now->left->ymin=now->ymin;
          now->left->ymax=now->ymax;

          now->right->zmin=now->value;
          now->right->zmax=now->zmax;
          now->right->xmin=now->xmin;
          now->right->xmax=now->xmax;
          now->right->ymin=now->ymin;
          now->right->ymax=now->ymax;
        }
        break; // end of split on z
      } // switch(axis) ended

      printf("axis = %d value = %.6lf\n", now->axis, now->value);

      if(continueSplit)
      {
        printf("next depth = %d\n", now->depth+1);
        now->left->depth  = now->depth+1;
        now->right->depth = now->depth+1;
        printf("depth set\n");

        queue.push_back(now->left);
        queue.push_back(now->right);
        printf("queue pushed\n");
      }
      printf("now queue size = %d \n", queue.size());
    }
  }

  __host__ __device__ void Print()
  {
    printf("CudaKdTree\n");
    if(root) root->Print();
    else printf("empty tree\n");
  }

  __host__ __device__ CudaVertex Hit(CudaRay ray, bool debuginfo = false)
  {
    if(root==nullptr)
    {
      return CudaVertex(false);
    }
    else
    {
      CudaVector<Node*> queue;
      queue.push_back(root);
      float global_min = 1e20;
      CudaVertex global_minp(false);
      while(queue.size()>0)
      {
        Node* now = queue.last();
        queue.remove(queue.size()-1);
        if(!now->Hit(ray, debuginfo))
        {
          continue;
        }
        if(now->left!=nullptr && now->right!=nullptr)
        {
          queue.push_back(now->left);
          queue.push_back(now->right);
        }
        else
        {
          float min=1e20;
          CudaVertex minp(false);
          for(int gid=0; gid<now->gl.size(); gid++)
          {
            CudaGeometry& geo = geolist[now->gl[gid]];
            CudaVertex result(false);
            ray.IntersectGeo(geo, result);
            if(result.valid)
            {
              float d=(result.p-ray.o).Length();
              if(d>EPS && d<min)
              {
                min=d;
                minp=result;
              }
            }
          }
          if(min>EPS && min<global_min)
          {
            global_min = min;
            global_minp = minp;
          }
        }
      }

      return global_minp;
    }

  }
}; // End of CudaKdTree

#define SampleNum (256)
#define RayTraceLength (4)
#define LightGeoSampleNum (1)

static CudaGeometry* dev_geos = nullptr;
static double* dev_hits = nullptr;
static double* dev_randNum = nullptr;
static int* dev_lightGeoList = nullptr;
static CudaKdTree* dev_kdtree = nullptr;

static int n = 0;
static double* hits = nullptr;
static int randN = 0;
static int* lightGeoList = nullptr;
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

__device__ void CudaMonteCarloSample(CudaGeometry* geolist, int n, int* lightGeoList, int ln, CudaVertex o, CudaRay i, double* randNum, int randN, CudaVec& result, int index, CudaKdTree* kdtree, curandState_t* randstate, bool debuginfo = false)
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

    float p = curand_uniform(randstate);

    if(p<=geolist[o.geo].reflectr)
    {
      ray.n = (currentIn.n-2*o.n*(currentIn.n.Dot(o.n))).Normalized();
      ray.d = currentIn.d;
    }
    else if(p<=geolist[o.geo].reflectr+geolist[o.geo].refractr)
    {
      if(fabs(currentIn.d-geolist[o.geo].ni)<EPS)
      {
        ray.d = 1.0;
      }
      else
      {
        ray.d = geolist[o.geo].ni;
      }

      if((currentIn.n+o.n).Length()<EPS)
      {
        // verticle
        ray.n = currentIn.n;
      }
      else
      {
        float sinn1 = sqrt(1-pow(currentIn.n.Dot(o.n), 2));
        float sinn2 = currentIn.d*sinn1/ray.d;
        if(sinn2>=1.0)
        {
          // total reflection
          ray.n = (currentIn.n-2*o.n*(currentIn.n.Dot(o.n))).Normalized();
          ray.d = currentIn.d;
        }
        else
        {
          CudaVec parallelAxis = currentIn.n.Vec3()-currentIn.n.Dot(o.n)*o.n.Vec3();
          parallelAxis = parallelAxis.Normalized();
          float cosn2 = sqrt(1-sinn2*sinn2);
          ray.n = CudaVec4(o.n.Vec3()*(-cosn2)+parallelAxis*sinn2, 0.0);
        }
      }
    }
    else
    {
      float sin2theta=curand_uniform(randstate); // sin^2(theta)
      float sintheta=sqrt(sin2theta);
      float phi = curand_uniform(randstate)*2*M_PI;
      if(debuginfo)
      {
        printf("%d : sin2theta = %.6lf, phi = %.6lf, index = %d\n", index, sin2theta, phi, index);
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

    ray.n.w=0; ray.n = ray.n.Normalized(); ray.o.w=1;

//    currentColor = currentColor*CudaMonteCarloHit(geolist, n, ray, o, haslight);

    double mind=1e20;
    CudaVertex minp(false);

    for(int i=0; i<n; i++)
    {
      CudaVertex hp;
      ray.IntersectGeo(geolist[i], hp);
      if(hp.valid)
      {
        if((hp.p-ray.o).Length()>EPS && (hp.p-ray.o).Length()<mind)
        {
          mind=(hp.p-ray.o).Length();
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
        if(r>2/fabs(ray.n.Dot(o.n))) r=2/fabs(ray.n.Dot(o.n));
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
        if(r>2) r=2;
        currentColor = currentColor * geolist[minp.geo].diffuse * fabs(ray.n.Dot(o.n)) * r;
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

    if(1-geolist[o.geo].reflectr-geolist[o.geo].refractr>EPS)
    {
      result = CudaVec(0, 0, 0);
      if(o.valid)
      {
        if(geolist[o.geo].reflectr>EPS)
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
          for(int j=0; j<LightGeoSampleNum; j++)
          {
            CudaVertex v = geolist[lightGeoList[ii]].Sample(randstate, randNum, randN);
            if(debuginfo)
            {
  //            printf("  sample vertex # %d\n", j);
              printf("  d = %.6lf, ", (v.p-o.p).Length()); v.p.Print(); printf("\n");
            }
            CudaRay ray; ray.o = o.p; ray.n = CudaVec4((v.p.Vec3()-o.p.Vec3()).Normalized(), 0.0);
            if(ray.n.Dot(o.n)<0) continue;
            bool visible = true;
            int lighthit = lightGeoList[ii];
            double d = (v.p-o.p).Length();
            for(int k=0; k<n; k++)
            {
              if(k==lightGeoList[ii]) continue;
              CudaVertex ir;
              ray.IntersectGeo(geolist[k], ir);
              if(ir.valid && (ir.p-o.p).Length()<d && (ir.p-o.p).Length()>EPS)
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

__global__ void CudaMonteCarloRender(CudaGeometry* geolist, int n, int* lightGeoList, int ln, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer, double* randNum, int randN, CudaKdTree* kdtree, bool debuginfo = false)
{
  __shared__ CudaVec tResults[SampleNum];

  if(n==0) return;
  int xx=blockIdx.x;
  int yy=blockIdx.y;
  int tid=threadIdx.x;
  curandState_t state;
//  if(xx==w/2 && yy==h/4)
//  {
//    for(int rid=0; rid<100; rid++)
//    {
//      printf("rand # %d = %.6lf\n", rid, curand_uniform(&state));
//    }
//  }
  tResults[tid] = CudaVec(0, 0, 0);

  if(xx<0 || xx>=w || yy<0 || yy>=h) return;
  int index = xx+yy*w;
  int seed = index*blockDim.x+threadIdx.x;
  curand_init(seed, seed, 0, &state);
  CudaRay ray = GetRay(xx, yy, w, h, camera, up, forward, right);

//  CudaLinkedList<double> list;
//  for(int i=0; i<20; i++)
//  {
//    double rand = curand_uniform(&state);
//    printf("rand # %d = %.6lf\n", i, rand);
//    list.push_back(rand);
//  }
//  for(int i=0; i<20; i++)
//  {
//    printf("list # %d = %.6lf\n", i, list.pop_head());
//  }
//  return;

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
        if(d>EPS && d<mind)
        {
          mind=d;
          minp=hp;
        }
      }
    }

    CudaVertex kdtreeresult;
//    if(seed == 0)
    {
//      kdtreeresult = kdtree->Hit(ray);
    }

//    if(kdtreeresult.valid == minp.valid && (!minp.valid || ((kdtreeresult.p-minp.p).Length()<EPS && kdtreeresult.geo==minp.geo)))
    {
      if(minp.valid)
      {
        CudaVec result;
        CudaMonteCarloSample(geolist, n, lightGeoList, ln, minp, ray, randNum, randN, result, seed, kdtree, &state, debuginfo);
        tResults[tid] = result;
      }
    }
//    else
    {
//      tResults[tid] = CudaVec(1.0, 0, 0);
    }
  }

  if(tid==0)
  {
    int count=1;
    for(int i=1; i<blockDim.x; i++)
    {
      if(tResults[tid].Length()<=EPS) continue;
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
        if(d>EPS && d<mind)
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
//      CudaMonteCarloSample(geolist, n, lightGeoList, ln, minp, ray, randNum, randN, result, index, debuginfo);
      buffer[index] = buffer[index] + result;
    }
  }

//  buffer[index] = buffer[index]/SampleNum;

  return;
}

__global__ void CudaBuildKdTree(CudaGeometry* geolist, int n, CudaKdTree* kdtree)
{
  printf("CudaBuildKdTree %d\n", n);
  if(n<=0) return;
  kdtree->SetTree(geolist, n);
//  kdtree->Print();
  printf("CudaBuildKdTree %d ended\n", n);
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

  if(dev_kdtree==nullptr)
  {
    gpuErrchk( cudaMalloc(&dev_kdtree, sizeof(CudaKdTree)));
    gpuErrchk( cudaMemset(dev_kdtree, 0, sizeof(CudaKdTree)));
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

//  CudaBuildKdTree<<<1, 1>>>(dev_geos, _n, dev_kdtree);
  fflush(stdout);
//  cudaDeviceSynchronize();
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
  gpuErrchk( cudaMalloc((void**)&dev_buffer, sizeof(CudaVec)*(w*h)));

  printf("start cuda render\n");
  printf("buffer.size=%d %d\n", sizeof(buffer), sizeof(buffer)/sizeof(CudaVec));
  fflush(stdout);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
//  for(int i=0; i<4; i++)
//  {
    CudaMonteCarloRender<<<dim3(w, h), SampleNum>>>(dev_geos, n, dev_lightGeoList, ln, w, h, camera, up, forward, right, dev_buffer, dev_randNum, randN, dev_kdtree);
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
  gpuErrchk( cudaMemcpy(buffer, dev_buffer, sizeof(CudaVec)*(w*h), cudaMemcpyDeviceToHost));
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

__global__ void debug_CudaVectorTest(int n, int* dev_int)
{
  CudaVector<int> test;
  for(int i=0; i<n; i++)
  {
    test.push_back(i*i);
  }
  for(int i=0; i<test.size(); i++)
  {
    printf("%d / %d : %d\n", i, test.size(), test[i]);
  }
}

extern "C" void CudaVectorTest()
{
#define TESTN (100)
  int array[TESTN];
  int* dev_array;
  cudaMalloc((void**)&dev_array, sizeof(int)*TESTN);
  printf("CudaVectorTest\n");
  fflush(stdout);
  debug_CudaVectorTest<<<1, 1>>>(TESTN, dev_array);
  fflush(stdout);
  cudaMemcpy(array, dev_array, sizeof(int)*TESTN, cudaMemcpyDeviceToHost);
}

__global__ void debug_CudaKdTreeTest(CudaGeometry* geolist, int n, CudaKdTree* kdtree)
{
  printf("debug_CudaKdTreeTest %d\n", n);
  if(n<=0) return;
  kdtree->SetTree(geolist, n);
  kdtree->Print();
}

extern "C" void CudaKdTreeTest(CudaGeometry* geolist, int n)
{
  printf("CudaKdTreeTest\n");
  fflush(stdout);
  CudaKdTree* kdtree;
  CudaKdTree output;
  cudaMalloc(&kdtree, sizeof(CudaKdTree));
  cudaMemset(kdtree, 0, sizeof(CudaKdTree));
  printf("read in\n");
  fflush(stdout);
  debug_CudaKdTreeTest<<<1, 1>>>(dev_geos, n, kdtree);
  fflush(stdout);
//  cudaMemcpy(&output, kdtree, sizeof(CudaKdTree), cudaMemcpyDeviceToHost);
  printf("CudaKdTreeTest ended\n");
  fflush(stdout);
}
