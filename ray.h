#ifndef RAY_H
#define RAY_H

#include <QVector3D>
#include "geometry.h"

class Ray
{
public:
  Ray(QVector3D oo = QVector3D(0, 0, 0), QVector3D nn = QVector3D(0, 0, -1));

  QVector3D o;
  QVector3D n;

  VertexInfo IntersectPlane(QVector4D pi);
  VertexInfo IntersectGeo(GI geo);
};

#endif // RAY_H
