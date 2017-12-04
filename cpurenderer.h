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
#include "daemodel.hpp"
#include "transform3d.h"
#include "camera3d.h"
#include "light.h"

#ifndef CGL_DEFINES
#define CGL_DEFINES

#endif

struct ColorPixel
{
  uint bytesPerPixel;
  uchar* bytes;

  ColorPixel(const uint& b=3) : bytes(nullptr), bytesPerPixel(b)
  {
    if(b>0)
    {
      b=new uchar[b];
    }
  }

  ~ColorPixel() { if(bytes) delete[] bytes; }

  void SetBytesPerPixel(const uint& b)
  {
   if(bytes)
   {
     delete[] bytes;
   }
   bytes = new uchar[b];
   bytesPerPixel = b;
  }

  uchar& r() { return bytes[0]; }
  uchar& g() { return bytes[1]; }
  uchar& b() { return bytes[2]; }
  uchar& a() { return bytes[3]; }
};

struct DepthFragment
{
  ColorPixel color;
  float depth;
  float ratio;
  bool opaque;
};

typedef QLinkedList<DepthFragment> DepthPixel;

class CPURenderer : public QObject
{
  Q_OBJECT
public:
  CPURenderer();

public slots:
  void Transform(QVector3D trans);
  void Rotate(QVector3D axis, double angle);
  void PushMatrix();
  void PopMatrix();
  void Resize(QSize w);
  QSize Size();
  uchar* RenderDaeScene(DaeModel* model);

signals:
  void OutputColorFrame(uchar* frame);

private:
  QSize size;

  QVector<Camera3D> cameras;
  QStack<Transform3D> transforms;
  QVector<Light> lights;

  DepthPixel* depthBuffer;
  uchar* stencilBuffer;

  QVector2D VertexShader(QVector3D input);
};

#endif // CPURENDERER_H
