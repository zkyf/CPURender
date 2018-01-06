#include "ray.h"

Ray::Ray(QVector4D oo, QVector4D nn)
{
  o=oo;
  n=nn;
}

QVector4D Ray::IntersectPlane(QVector4D pi)
{
  double pn = QVector4D::dotProduct(pi, n);
  if(fabs(pn)<1e-3)
  {
    return QVector4D(0, 0, 0, 0);
  }
  double r = -QVector4D::dotProduct(pi, o)/pn;
  if(r<0) return QVector4D(0, 0, 0, 0);
  return r*n+o;
}

VertexInfo Ray::IntersectGeo(GI geo)
{
  return VertexInfo();
}
