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
#include "geometry.h"

#include <opencv2/opencv.hpp>
using namespace cv;

#ifndef CGL_DEFINES
#define CGL_DEFINES

#endif

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
  double dx;
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
  void ClearGeometry() { input.clear(); textures.clear(); }

  // textures
  int AddTexture(QImage text);
  ColorPixel Sample(int tid, double u, double v, bool bi=false);

  uchar* Render();

  int ToScreenY(double y);
  int ToScreenX(double x);
  double ToProjY(int y);
  double ToProjX(int x);

  DepthPixel DepthPixelAt(int x, int y);
signals:
  void OutputColorFrame(uchar* frame);

public:
  QVector<Light> lights;

private:
  QSize size;

  double nearPlane;
  double farPlane;
  QMatrix4x4 projection;

  Camera3D camera;
  Transform3D transform;

  QVector<Geometry> geos;
  QVector<Geometry> input;
  QVector<QImage> textures;
  ColorBuffer colorBuffer;
  DepthBuffer depthBuffer;
  uchar* stencilBuffer;

  VertexInfo VertexShader(VertexInfo v);
  void GeometryShader(Geometry& geo);
  void FragmentShader(DepthFragment& frag);
  void Clip(QVector4D A, bool dir, QVector<QVector4D> g, GI i, bool& dirty, bool pers=true);
};

QVector3D operator*(const QMatrix3x3& m, const QVector3D& x);

#endif // CPURENDERER_H
