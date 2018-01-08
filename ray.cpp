//#include "ray.h"

//Line::Line(QVector4D oo, QVector4D nn)
//{
//  o=oo;
//  n=nn;
//}

//bool Line::SamePlane(const Line &l)
//{

//}

//Point Line::Intersect(const Plane &pi)
//{
//  double pn = QVector4D::dotProduct(pi, n);
//  if(fabs(pn)<1e-3)
//  {
//    return QVector4D(0, 0, 0, 0);
//  }
//  double r = -QVector4D::dotProduct(pi, o)/pn;
//  return r*n+o;
//}

//Point Line::Intersect(const Line &l)
//{
//  if(!SamePlane(l)) return Point(0, 0, 0, 0);
//  if(Same(l)) return o;

//}

//Ray::Ray(QVector4D oo, QVector4D nn)
//{
//  o=oo;
//  n=nn;
//  n.setW(0);
//}

//VertexInfo Ray::IntersectGeo(GI geo)
//{
//  if(geo->vecs.size()<3 || !geo->IsPlane())
//    return VertexInfo();

//  // ray intersection using tp coordinate
//  QVector4D p = Intersect(geo->Plane());
//  if((geo->vecs[0].tp-p).length()<1e-3) return geo->vecs[0];

//  QVector4D cp = p-geo->vecs[0].tp;
//  QVector4D cpn = cp.normalized();
//  QVector4D ca = geo->vecs[1].tp - geo->vecs[0].tp;
//  QVector4D cb = geo->vecs[2].tp - geo->vecs[0].tp;
//  QVector4D cd = QVector4D::dotProduct(ca, cpn)*cpn;
//  QVector4D ce = QVector4D::dotProduct(cb, cpn)*cpn;
//  double rb = (ca-cd).length()/(cb-cd).length();
//  rb = rb/(1+rb);
//  double ra = 1-rb;
//  QVector4D f = rb*geo->vecs[2].tp+ra*geo->vecs[1].tp;
//  double rc = 1-cp.length()/(f-geo->vecs[0].tp).length();
//  ra *= (1-rc); rb *= (1-rc);
//  if(ra<0 || rb<0 || rc<0) return VertexInfo();
//  else
//  {
//    VertexInfo vf = VertexInfo::Intersect(geo->vecs[1], geo->vecs[2], ra, false);
//    VertexInfo vp = VertexInfo::Intersect(geo->vecs[0], vf, rc, false);
//    return vp;
//  }
//}
