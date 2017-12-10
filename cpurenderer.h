#ifndef CPURENDERER_H
#define CPURENDERER_H

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

#ifndef CGL_DEFINES
#define CGL_DEFINES

#endif

struct Geometry
{
  QVector<QVector3D> vecs;
  QVector<QVector3D> norms;
  QVector<QVector2D> texcoords;

  QVector3D ambient;
  QVector3D diffuse;
  QVector3D specular;

  float top;
  float bottom;

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

  bool operator>(Geometry& b) { return Top()>b.Top(); }
  bool operator<(Geometry& b) { return Top()<b.Top(); }
  bool operator>=(Geometry& b) { return Top()>=b.Top(); }
  bool operator<=(Geometry& b) { return Top()<=b.Top(); }
  bool operator==(Geometry& b) { return Top()==b.Top(); }
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
    buffer = new ColorPixel[size.width(), size.height()];
  }
  void Clear(ColorPixel p = ColorPixel())
  {
    for(int i=0; i<size.width()*size.height(); i++)
    {
      buffer[i]=p;
    }
  }
};

struct DepthFragment
{
  ColorPixel color;
  float depth;
  float ratio;
  bool opaque;

  bool operator>(const DepthFragment& b) { return depth>b.depth; }
  bool operator>=(const DepthFragment& b) { return depth>=b.depth; }
  bool operator<(const DepthFragment& b) { return depth<b.depth; }
  bool operator<=(const DepthFragment& b) { return depth<=b.depth; }
  bool operator==(const DepthFragment& b) { return depth==b.depth; }
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
};

struct ScanlinePoint : public QVector3D
{
  GI geo;
  int hp;

  bool operator<(ScanlinePoint& b) { return xp<b.xp; }
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
  void Resize(QSize w);
  QSize Size();
  Light* AddLight(Light light);
  void RemoveLight(Light* light);

  ColorPixel* Render();

  int ToScreenY(float y);
  int ToScreenX(float x);
  int ToProjY(int y);
  int ToProjX(int x);
signals:
  void OutputColorFrame(uchar* frame);

private:
  QSize size;

  float nearPlane;
  float farPlane;

  Camera3D camera;
  Transform3D transform;
  QLinkedList<Light> lights;

  QVector<Geometry> geos;
  ColorBuffer colorBuffer;
  DepthBuffer depthBuffer;
  uchar* stencilBuffer;

  void VertexShader(QVector3D& p, QVector3D& n, QVector2D& tc);
  void GeometryShader(Geometry& geo);
  void Clip();
  void FragmentShader(QVector2D pos, QVector2D texcoord);
};

#endif // CPURENDERER_H
