#include "ray.h"

Ray::Ray(QVector3D oo = QVector3D(0, 0, 0), QVector3D nn = QVector3D(0, 0, -1))
{
  o=oo;
  n=nn;
}

VertexInfo Ray::IntersectPlane(QVector4D pi)
{
}

VertexInfo Ray::IntersectGeo(GI &geo)
{

}
