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

struct VertexInfo
{
  QVector3D pos;
  QVector3D projpos;
  QVector3D norm;
  QVector3D transformed_pos;
  QVector2D texcoords;
};

struct Geometry
{
  QString name;
  QVector<QVector3D> pos;
  QVector<QVector3D> vecs;
  QVector<QVector3D> norms;
  QVector<QVector2D> texcoords;

  QVector3D ambient;
  QVector3D diffuse;
  QVector3D specular;

  float top;
  float bottom;
  float dz;
  QVector<float> dy;

  float Top()
  {
    top = -1e20;
    for(int i=0; i<vecs.size(); i++)
    {
      if(vecs[i].y()>top)
      {
        top=vecs[i].y();
      }
    }

    return top;
  }

  float Bottom()
  {
    bottom = 1e20;
    for(int i=0; i<vecs.size(); i++)
    {
      if(vecs[i].y()<bottom)
      {
        bottom=vecs[i].y();
      }
    }

    return bottom;
  }

  friend bool operator> (const Geometry& a, const Geometry& b);
  friend bool operator< (const Geometry& a, const Geometry& b);
  friend bool operator>=(const Geometry& a, const Geometry& b);
  friend bool operator<=(const Geometry& a, const Geometry& b);
  friend bool operator==(const Geometry& a, const Geometry& b);
};

typedef QVector<Geometry>::iterator GI;

struct ColorPixel
{
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
};

struct ColorBuffer
{
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

struct DepthFragment
{
  GI geo;
  ColorPixel color;
  QVector3D normal;
  QVector3D pos;

  friend bool operator> (const DepthFragment& a, const DepthFragment& b);
  friend bool operator>=(const DepthFragment& a, const DepthFragment& b);
  friend bool operator< (const DepthFragment& a, const DepthFragment& b);
  friend bool operator<=(const DepthFragment& a, const DepthFragment& b);
  friend bool operator==(const DepthFragment& a, const DepthFragment& b);
};

struct DepthPixel
{
  QVector<DepthFragment> chain;
};

struct DepthBuffer
{
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

struct ScanlinePoint : public QVector3D
{
  GI geo;
  int hp;
  int prev;
  int next;
  float dx;

  ScanlinePoint(QVector3D p=QVector3D(0, 0, 0)):QVector3D(p) {}
};

typedef QVector<ScanlinePoint> Scanline;

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

  // camera config
  void CamTranslate(QVector3D trans);
  QVector3D CamTranslation();
  void CamRotate(QVector3D axis, double angle);
  QQuaternion CamRotation();

  // operations
  void AddGeometry(const Geometry& geo);
  void Resize(QSize w);
  QSize Size();
  void AddLight(Light light);

  uchar* Render();

  int ToScreenY(float y);
  int ToScreenX(float x);
  float ToProjY(int y);
  float ToProjX(int x);
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

  QVector3D VertexShader(QVector3D& p, QVector3D& n, QVector2D& tc);
  void GeometryShader(Geometry& geo);
  void FragmentShader(DepthFragment& frag);
};

#endif // CPURENDERER_H
