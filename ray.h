#ifndef RAY_H
#define RAY_H

#include <QDebug>
#include <QVector3D>
#include <QVector4D>
#include "geometry.h"

typedef QVector4D Point;

struct Line;

struct Plane : public QVector4D
{
  Plane();
  Plane(const QVector4D& v);
  bool On(const Point& p);
  bool On(const Line& l);
};

struct Line
{
  QVector4D o;
  QVector4D n;

  Line(QVector4D oo = QVector4D(0, 0, 0, 1), QVector4D nn = QVector4D(0, 0, -1, 0));

  void SetFromTo(Point a, Point b);

  bool On(const Point& p);

  bool Verticle(const Line& l);
  bool Parallel(const Line& l);
  bool Same(const Line& l);
  bool SamePlane(const Line& l);

  virtual Point Intersect(const Plane& pi);
  virtual Point Intersect(const Line& l);
};

struct Ray : public Line
{
public:
  ColorPixel color;
  float ni=1.0;

  Ray(QVector4D oo = QVector4D(0, 0, 0, 1), QVector4D nn = QVector4D(0, 0, -1, 0));

  Point Intersect(const Plane &pi) override;
  VertexInfo IntersectGeometry(Geometry geo, bool debuginfo=false);
  VertexInfo IntersectGeo(GI geo, bool debuginfo=false);
  Ray Reflect(Point p, QVector4D nn);
  Ray Refract(Point P, QVector4D nn, double ratio);
};

QDebug& operator<<(QDebug& s, const Ray& ray);

#endif // RAY_H
