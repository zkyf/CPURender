#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <QDebug>
#include <QImage>
#include <QVector>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <time.h>

class Geometry;
typedef QVector<Geometry>::iterator GI;

class ColorPixel
{
public:
  double r;
  double g;
  double b;
  double a;

  ColorPixel(double rr=0, double gg=0, double bb=0, double aa=0);
  ColorPixel(QVector3D c);
  ColorPixel(QVector4D c);

  void Clamp();
  void LinearBlend(const ColorPixel& c);
  float Strength() const;

  ColorPixel operator=(const QVector3D& c);
  ColorPixel operator=(const QVector4D& c);
  ColorPixel operator+(ColorPixel& c);
  ColorPixel operator+=(ColorPixel& c);
  ColorPixel operator*(const double& c);
  ColorPixel operator/(const double& c);
  ColorPixel operator*(const ColorPixel& c);

  friend QDebug& operator<<(QDebug& s, const ColorPixel& p);
};
QDebug& operator<<(QDebug& s, const ColorPixel& p);

class VertexInfo
{
public:
  // original position in the world
  QVector3D p;
  // normal
  QVector3D n;
  // texture co-ordinate
  QVector2D tc;
  // texture projected
  QVector2D ptc;
  // transformed position in Ether frame
  QVector3D wp;

  // These 2 variables are used in vertex processing
  // transformed position in camera frame
  QVector3D tp;
  // projected posotion in perspective frame
  QVector4D pp;

  GI geo;

  bool valid;

public:
  VertexInfo(bool v= true, QVector3D _p = QVector3D(0, 0, 0), QVector3D _n=QVector3D(0.0, 0.0, 0.0), QVector2D _tc=QVector2D(0.0, 0.0));

  static VertexInfo Intersect(VertexInfo a, VertexInfo b, double r, bool pers=true);
};

typedef QVector<VertexInfo>::iterator VI;

class Geometry
{
public:
  QString name;

  QVector<VertexInfo> vecs;

  double top;
  double bottom;
  double dz;
  double dzy;
  double ns;
  int text=-1;
  int stext=-1;
  bool inout;
  int id;

  // materials
  ColorPixel ambient;
  ColorPixel diffuse;
  ColorPixel specular;
  ColorPixel emission;

  bool diff = true;
  bool tran = false;
  bool spec = false;

  float refractr = 0.0;
  float reflectr = 0.0;

  double ni=1.0;

  QImage dt;

  void AddPoints(QVector<QVector3D> vs);
  double Top();
  double Bottom();
  void SetNormal();
  QVector3D Normal();
  bool IsPlane();
  bool IsInside(QVector3D p);
  QVector4D Plane();
  VertexInfo Sample(bool debuginfo=false);
  GI gi;

  friend bool operator> (const Geometry& a, const Geometry& b);
  friend bool operator< (const Geometry& a, const Geometry& b);
  friend bool operator>=(const Geometry& a, const Geometry& b);
  friend bool operator<=(const Geometry& a, const Geometry& b);
  friend bool operator==(const Geometry& a, const Geometry& b);
  friend QDebug& operator<<(QDebug& s, const Geometry& g);
};
bool operator> (const Geometry& a, const Geometry& b);
bool operator< (const Geometry& a, const Geometry& b);
bool operator>=(const Geometry& a, const Geometry& b);
bool operator<=(const Geometry& a, const Geometry& b);
bool operator==(const Geometry& a, const Geometry& b);
QDebug& operator<<(QDebug& s, const Geometry& g);

#endif // GEOMETRY_H
