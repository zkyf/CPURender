#ifndef CPURENDERER_H
#define CPURENDERER_H

#include <QStack>;
#include <QVector>
#include <QVector3D>
#include <QVector2D>
#include <QMatrix4x4>
#include "daemodel.hpp"
#include "transform3d.h"
#include "camera3d.h"
#include "light.h"

class CPURenderer
{
public:
  CPURenderer();

  void Transform(QVector3D trans);
  void Rotate(QVector3D axis, double angle);
  void PushMatrix();
  void PopMatrix();

private:
  QVector<Camera3D> cameras;
  QStack<Transform3D> transforms;
  QVector<Light> lights;
};

#endif // CPURENDERER_H
