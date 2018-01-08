#include "ray.h"

Line::Line(QVector4D oo, QVector4D nn)
{
  o=oo;
  n=nn;
}

bool Line::SamePlane(const Line &l)
{

}

Point Line::Intersect(const Plane &pi)
{
  double pn = QVector4D::dotProduct(pi, n);
  if(fabs(pn)<1e-3)
  {
    return QVector4D(0, 0, 0, 0);
  }
  double r = -QVector4D::dotProduct(pi, o)/pn;
  return r*n+o;
}

Point Line::Intersect(const Line &l)
{
  QVector4D d = l.o-o;
  QVector3D nof = QVector3D::crossProduct(QVector3D(n), QVector3D(l.n));
  if(QVector3D::dotProduct(QVector3D(d), nof));
}

Ray::Ray(QVector4D oo, QVector4D nn)
{
  o=oo;
  n=nn;
  n.setW(0);
}

VertexInfo Ray::IntersectGeo(GI geo)
{
  if(geo->vecs.size()<3 || !geo->IsPlane())
    return VertexInfo();

  // ray intersection using tp coordinate
  QVector4D p = Intersect(geo->Plane());
  if((geo->vecs[0].tp-p).length()<1e-3) return geo->vecs[0];


}
