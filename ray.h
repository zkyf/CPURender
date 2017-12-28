#ifndef RAY_H
#define RAY_H

#include <QVector3D>
#include "geometry.h"

class Ray
{
public:
  Ray(QVector4D oo = QVector4D(0, 0, 0, 1), QVector4D nn = QVector4D(0, 0, -1, 0));

  QVector4D o;
  QVector4D n;

  QVector4D IntersectPlane(QVector4D pi);
  VertexInfo IntersectGeo(GI geo);
};

#endif // RAY_H
