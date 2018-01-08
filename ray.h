//#ifndef RAY_H
//#define RAY_H

//#include <QVector3D>
//#include <QVector4D>
//#include "geometry.h"

//typedef QVector4D Point;

//struct Plane : public QVector4D
//{
//  Plane();
//  Plane(const QVector4D& v);
//  bool On(const Point& p);
//  bool On(const Line& l);
//};

//struct Line
//{
//  QVector4D o;
//  QVector4D n;

//  Line(QVector4D oo = QVector4D(0, 0, 0, 1), QVector4D nn = QVector4D(0, 0, -1, 0));

//  void SetFromTo(Point a, Point b);

//  bool On(const Point& p);

//  bool Verticle(const Line& l);
//  bool Parallel(const Line& l);
//  bool Same(const Line& l);
//  bool SamePlane(const Line& l);

//  Point Intersect(const Plane& pi);
//  Point Intersect(const Line& l);
//};

//struct Ray : public Line
//{
//public:
//  Ray(QVector4D oo = QVector4D(0, 0, 0, 1), QVector4D nn = QVector4D(0, 0, -1, 0));

//  VertexInfo IntersectGeo(GI geo);
//};

//#endif // RAY_H
