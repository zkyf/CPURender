#ifndef CPURENDERER_H
#define CPURENDERER_H

#include <QDebug>
#include <QLinkedList>
#include <QMatrix4x4>
#include <QObject>
#include <QSize>
#include <QStack>
#include <QVector>
#include <QVector3D>
#include <QVector2D>
#include <QtAlgorithms>
#include "transform3d.h"
#include "camera3d.h"
#include "light.h"
#include "simplemesh.h"

#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef CGL_DEFINES
#define CGL_DEFINES

#endif

class ColorPixel
{
public:
  float r;
  float g;
  float b;
  float a;

  ColorPixel(float rr=0, float gg=0, float bb=0, float aa=0) :
    r(rr), g(gg), b(bb), a(aa) {}
  ColorPixel(QVector3D c) { r=c.x(); g=c.y(); b=c.z(); a=1.0; }
  ColorPixel operator=(const QVector3D& c) { r=c.x(); g=c.y(); b=c.z(); a=1.0; return *this; }
  ColorPixel(QVector4D c) { r=c.x(); g=c.y(); b=c.z(); a=c.w(); }
  ColorPixel operator=(const QVector4D& c) { r=c.x(); g=c.y(); b=c.z(); a=c.w(); return *this; }

  void Clamp()
  {
    if(r<0) r=0; if(r>1) r=1;
    if(g<0) g=0; if(g>1) g=1;
    if(b<0) b=0; if(b>1) b=1;
    if(a<0) a=0; if(a>1) a=1;
  }

  ColorPixel operator+(ColorPixel& c)
  {
    float r1 = a/(a+c.a*(1-a));
    float r2 = 1-r1;
    return ColorPixel(
          r*r1+c.r*r2,
          g*r1+c.g*r2,
          b*r1+c.b*r2,
          a+c.a*(1-a));
  }

  ColorPixel operator+=(ColorPixel& c)
  {
    *this = *this+c;
    return *this;
  }

  ColorPixel operator*(const float& c)
  {
    return ColorPixel(r*c, g*c, b*c, a);
  }
};
QDebug& operator<<(QDebug& s, const ColorPixel& p);

class VertexInfo
{
public:
  // original position in the world
  QVector3D p;
  // normal
  QVector3D n;
  // texture co-ordinate
  QVector2D tc;
  // transformed position in Ether frame
  QVector3D wp;

  // These 2 variables are used in vertex processing
  // transformed position in camera frame
  QVector3D tp;
  // projected posotion in perspective frame
  QVector3D pp;

public:
  VertexInfo() {}
  VertexInfo(QVector3D _p, QVector3D _n=QVector3D(0.0, 0.0, 0.0), QVector2D _tc=QVector2D(0.0, 0.0)) : p(_p), n(_n), tc(_tc) {}

  friend VertexInfo operator*(const VertexInfo& v, const float& r);
  friend VertexInfo operator*(const float& r, const VertexInfo& v);
  friend VertexInfo operator+(const VertexInfo& a, const VertexInfo& b);
  friend VertexInfo operator-(const VertexInfo& a, const VertexInfo& b);
};
VertexInfo operator*(const VertexInfo& v, const float& r);
VertexInfo operator*(const float& r, const VertexInfo& v);
VertexInfo operator+(const VertexInfo& a, const VertexInfo& b);
VertexInfo operator-(const VertexInfo& a, const VertexInfo& b);

class Geometry
{
public:
  QString name;

  QVector<VertexInfo> vecs;

  ColorPixel ambient;
  ColorPixel diffuse;
  ColorPixel specular;

  float top;
  float bottom;
  float dz;
  float dzy;
  bool inout;

  QImage dt;

  void AddPoints(QVector<QVector3D> vs)
  {
    for(int i=0; i<vs.size(); i++)
    {
      vecs.push_back(VertexInfo(vs[i]));
    }
  }

  float Top()
  {
    top = -1e20;
    for(int i=0; i<vecs.size(); i++)
    {
      if(vecs[i].pp.y()>top)
      {
        top=vecs[i].pp.y();
      }
    }

    return top;
  }

  float Bottom()
  {
    bottom = 1e20;
    for(int i=0; i<vecs.size(); i++)
    {
      if(vecs[i].pp.y()<bottom)
      {
        bottom=vecs[i].pp.y();
      }
    }

    return bottom;
  }

  void SetNormal()
  {
//    qDebug() << "vecs.size=" << vecs.size();
    if(vecs.size()>=3)
    {
      QVector3D v0=vecs[0].p;
      QVector3D v1=vecs[1].p;
      QVector3D v2=vecs[2].p;
//      qDebug() << "v0=" << v0 << "v1=" << v1 << "v2=" << v2;
      QVector3D n=QVector3D::crossProduct(v2-v1, v0-v1).normalized();
      for(int i=0; i<vecs.size(); i++)
      {
        vecs[i].n=n;
      }
    }
  }

  friend bool operator> (const Geometry& a, const Geometry& b);
  friend bool operator< (const Geometry& a, const Geometry& b);
  friend bool operator>=(const Geometry& a, const Geometry& b);
  friend bool operator<=(const Geometry& a, const Geometry& b);
  friend bool operator==(const Geometry& a, const Geometry& b);

  friend QDebug& operator<<(QDebug& s, const Geometry& g);
};
bool operator> (const Geometry& a, const Geometry& b);
bool operator< (const Geometry& a, const Geometry& b);
bool operator>=(const Geometry& a, const Geometry& b);
bool operator<=(const Geometry& a, const Geometry& b);
bool operator==(const Geometry& a, const Geometry& b);
QDebug& operator<<(QDebug& s, const Geometry& g);

typedef QVector<Geometry>::iterator GI;

class ColorBuffer
{
public:
  QSize size;
  ColorPixel* buffer;

  ColorBuffer(QSize s) : size(s), buffer(nullptr)
  {
    buffer = new ColorPixel[s.width()*s.height()];
  }
  ~ColorBuffer() { delete[] buffer; }
  void Resize(QSize s)
  {
    size = s;
    delete[] buffer;
    buffer = new ColorPixel[size.width()*size.height()];
  }

  void Clear(ColorPixel p = ColorPixel())
  {
    for(int i=0; i<size.width()*size.height(); i++)
    {
      buffer[i]=p;
    }
  }

  uchar* ToUcharArray()
  {
    uchar* result = new uchar[size.height()*size.width()*3];
    memset(result, 0, sizeof(uchar)*size.width()*size.height()*3);
    for(int i=0; i<size.width()*size.height(); i++)
    {
      int tr=i*3;
      int tg=i*3+1;
      int tb=i*3+2;
      buffer[i].Clamp();
      result[tr]=255*buffer[i].r;
      result[tg]=255*buffer[i].g;
      result[tb]=255*buffer[i].b;
    }
    return result;
  }
};

class DepthFragment
{
public:
  int index;
  GI geo;
  ColorPixel color;
  QVector3D pos;
  VertexInfo v;
  QVector3D lightDir;

  friend bool operator> (const DepthFragment& a, const DepthFragment& b);
  friend bool operator>=(const DepthFragment& a, const DepthFragment& b);
  friend bool operator< (const DepthFragment& a, const DepthFragment& b);
  friend bool operator<=(const DepthFragment& a, const DepthFragment& b);
  friend bool operator==(const DepthFragment& a, const DepthFragment& b);
};
bool operator> (const DepthFragment& a, const DepthFragment& b);
bool operator>=(const DepthFragment& a, const DepthFragment& b);
bool operator< (const DepthFragment& a, const DepthFragment& b);
bool operator<=(const DepthFragment& a, const DepthFragment& b);
bool operator==(const DepthFragment& a, const DepthFragment& b);
QDebug& operator<<(QDebug& s, const DepthFragment& f);

class DepthPixel
{
public:
  QVector<DepthFragment> chain;
};
QDebug& operator<<(QDebug& s, const DepthPixel& d);

class DepthBuffer
{
public:
  DepthPixel* buffer;
  QSize size;

  DepthBuffer(QSize s) : size(s), buffer(nullptr)
  {
    buffer = new DepthPixel[size.width()*size.height()];
  }
  ~DepthBuffer() { delete[] buffer; }
  void Clear()
  {
    for(int i=0; i<size.width()*size.height(); i++)
    {
      buffer[i].chain.clear();
    }
  }

  void Resize(QSize s)
  {
    for(int i=0; i<size.width()*size.height(); i++)
    {
      buffer[i].chain.clear();
    }
    size = s;
    delete[] buffer;
    buffer = new DepthPixel[s.width()*s.height()];
  }
};

class EdgeListItem
{
public:
  GI geo;
  int tid; // id of top vertex;
  int bid; // id of bottom vertex;

  friend bool operator<(const EdgeListItem& a, const EdgeListItem& b);
};
bool operator< (const EdgeListItem& a, const EdgeListItem& b);

class ScanlinePoint : public QVector3D
{
public:
  GI geo;
  int prev;
  int next;
  int e;
  float dx;
  VertexInfo v;

  ScanlinePoint(QVector3D p=QVector3D(0, 0, 0)):QVector3D(p) {}
};
QDebug& operator<<(QDebug& s, const ScanlinePoint& p);

typedef QVector<ScanlinePoint> Scanline;
QDebug& operator<<(QDebug& s, const Scanline& line);

class CPURenderer : public QObject
{
  Q_OBJECT
public:
  CPURenderer(QSize s=QSize(640, 480));
  ~CPURenderer();

public slots:
  // world transformation
  void WorldTranslate(QVector3D trans);
  QVector3D WorldTranslation();
  void WorldRotate(QVector3D axis, double angle);
  QQuaternion WorldRotation();
  void ResetWorld() { transform.setTranslation(0, 0, 0); transform.setRotation(0, 1, 0, 0);}

  // camera config
  void CamTranslate(QVector3D trans);
  QVector3D CamTranslation();
  void CamRotate(QVector3D axis, double angle);
  QQuaternion CamRotation();
  QVector3D CamUp() { return camera.up(); }
  QVector3D CamRight() { return camera.right(); }
  QVector3D CamForward() { return camera.forward(); }
  void ResetCam() { camera.setTranslation(0, 0, 0); camera.setRotation(0, 1, 0, 0); }

  // operations
  void AddGeometry(const Geometry& geo);
  void Resize(QSize w);
  QSize Size();
  void AddLight(Light light);
  void ClearGeometry() { input.clear(); }

  uchar* Render();

  int ToScreenY(float y);
  int ToScreenX(float x);
  float ToProjY(int y);
  float ToProjX(int x);

  DepthPixel DepthPixelAt(int x, int y);
signals:
  void OutputColorFrame(uchar* frame);

private:
  QSize size;

  float nearPlane;
  float farPlane;
  QMatrix4x4 projection;

  Camera3D camera;
  Transform3D transform;
  QVector<Light> lights;

  QVector<Geometry> geos;
  QVector<Geometry> input;
  ColorBuffer colorBuffer;
  DepthBuffer depthBuffer;
  uchar* stencilBuffer;

  VertexInfo VertexShader(VertexInfo v);
  void GeometryShader(Geometry& geo);
  void FragmentShader(DepthFragment& frag);
  void Clip(QVector4D A, bool dir, QVector<QVector3D> g, GI i, bool& dirty);
};

QVector3D operator*(const QMatrix3x3& m, const QVector3D& x);

#endif // CPURENDERER_H
