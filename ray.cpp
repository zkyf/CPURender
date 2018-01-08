#include "ray.h"

Plane::Plane() {}
Plane::Plane(const QVector4D &v) : QVector4D(v) {}

bool Plane::On(const Point &p)
{
  if(fabs(QVector4D::dotProduct(p, *this))<1e-3) return true;
  else return false;
}

bool Plane::On(const Line& l)
{
  if(!On(l.o)) return false;
  if(fabs(QVector4D::dotProduct(*this, l.n))<1e-3) return true;
  else return false;
}

Line::Line(QVector4D oo, QVector4D nn)
{
  o=oo;
  n=nn;
}

void Line::SetFromTo(Point a, Point b)
{
  o=a;
  n=(b-a).normalized();
}

bool Line::On(const Point &p)
{
  if(fabs(fabs(QVector4D::dotProduct(p-o, n))-1)<1e-3) return true;
  else return false;
}

bool Line::Verticle(const Line &l)
{
  return fabs(QVector4D::dotProduct(l.n, n))<1e-3;
}

bool Line::Parallel(const Line &l)
{
  return fabs(fabs(QVector4D::dotProduct(l.n, n))-1)<1e-3;
}

bool Line::Same(const Line &l)
{
  if(!Parallel(l)) return false;
  return fabs(QVector3D::crossProduct(QVector3D(n), QVector3D(l.o-o)).length())<1e-3;
}

bool Line::SamePlane(const Line &l)
{
  if(Parallel(l)) return true;
  QVector3D nn = QVector3D::crossProduct(QVector3D(l.n), QVector3D(n));
  return fabs(QVector3D::dotProduct(nn, QVector3D(l.o-o)))<1e-3;
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
  if(!SamePlane(l)) return Point(0, 0, 0, 0);
  if(Same(l)) return o;
}

Ray::Ray(QVector4D oo, QVector4D nn)
{
  o=oo;
  n=nn;
  n.setW(0);
}

Point Ray::Intersect(const Plane &pi)
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
  if(geo->vecs.size()<3 || !geo->IsPlane())
    return VertexInfo();

  // ray intersection using tp coordinate
  QVector4D p = Intersect(geo->Plane());
  if((geo->vecs[0].tp-p).length()<1e-3) return geo->vecs[0];

  QVector4D cp = p-geo->vecs[0].tp;
  QVector4D cpn = cp.normalized();
  QVector4D ca = geo->vecs[1].tp - geo->vecs[0].tp;
  QVector4D cb = geo->vecs[2].tp - geo->vecs[0].tp;
  QVector4D cd = QVector4D::dotProduct(ca, cpn)*cpn;
  QVector4D ce = QVector4D::dotProduct(cb, cpn)*cpn;
  double rb = (ca-cd).length()/(cb-cd).length();
  rb = rb/(1+rb);
  double ra = 1-rb;
  QVector4D f = rb*geo->vecs[2].tp+ra*geo->vecs[1].tp;
  double rc = 1-cp.length()/(f-geo->vecs[0].tp).length();
  ra *= (1-rc); rb *= (1-rc);
  if(ra<0 || rb<0 || rc<0) return VertexInfo();
  else
  {
    VertexInfo vf = VertexInfo::Intersect(geo->vecs[1], geo->vecs[2], ra, false);
    VertexInfo vp = VertexInfo::Intersect(geo->vecs[0], vf, rc, false);
    return vp;
  }
}

Ray Ray::Reflect(Point p, QVector4D nn)
{
  Ray result;
  result.o = p;
  result.n = n-2*QVector4D::dotProduct(n, nn)*nn;
  return result;
}
