#include "geometry.h"

/// class Color Pixel

ColorPixel::ColorPixel(double rr, double gg, double bb, double aa)
  : r(rr), g(gg), b(bb), a(aa) {}
ColorPixel::ColorPixel(QVector3D c) { r=c.x(); g=c.y(); b=c.z(); a=1.0; }
ColorPixel::ColorPixel(QVector4D c) { r=c.x(); g=c.y(); b=c.z(); a=c.w(); }

void ColorPixel::Clamp()
{
  if(r<0) r=0; if(r>1) r=1;
  if(g<0) g=0; if(g>1) g=1;
  if(b<0) b=0; if(b>1) b=1;
  if(a<0) a=0; if(a>1) a=1;
}

ColorPixel ColorPixel::operator=(const QVector3D& c)
{
  r=c.x();
  g=c.y();
  b=c.z();
  a=1.0;
  return *this;
}

ColorPixel ColorPixel::operator=(const QVector4D& c)
{
  r=c.x();
  g=c.y();
  b=c.z();
  a=c.w();
  return *this;
}

ColorPixel ColorPixel::operator +(ColorPixel& c)
{
  double r1 = a/(a+c.a*(1-a));
  double r2 = 1-r1;
  return ColorPixel(
        r*r1+c.r*r2,
        g*r1+c.g*r2,
        b*r1+c.b*r2,
        a+c.a*(1-a));
}

ColorPixel ColorPixel::operator +=(ColorPixel& c)
{
  *this = *this+c;
  return *this;
}

ColorPixel ColorPixel::operator *(const double& c)
{
  return ColorPixel(r*c, g*c, b*c, a);
}

ColorPixel ColorPixel::operator *(const ColorPixel& c)
{
  return ColorPixel(r*c.r, g*c.g, b*c.b, a*c.a);
}

QDebug& operator<<(QDebug& s, const ColorPixel& p)
{
  s << "(" << p.r << ", " << p.g << ", " << p.b << ", " << p.a << ")";
  return s;
}

/// Class VertexInfo
VertexInfo::VertexInfo(bool v, QVector3D _p, QVector3D _n, QVector2D _tc)
 : p(_p), n(_n), tc(_tc), valid(v) {}
VertexInfo VertexInfo::Intersect(VertexInfo a, VertexInfo b, double r, bool pers)
{
  VertexInfo result;
  if(pers)
  {
    // interpolation under perspective frame, so interpolate using 1/z (default)
    result.p = r*a.p+(1-r)*b.p;
    result.n = r*a.n+(1-r)*b.n;
//    result.wp = r*a.wp+(1-r)*b.wp;
    result.tp = r*a.tp+(1-r)*b.tp;
    result.pp = r*a.pp+(1-r)*b.pp;

    result.ptc=r*a.ptc+(1-r)*b.ptc;
    result.tc=result.ptc/result.pp.w();
//    result.tc = (a.tc*a.pp.w()*r+(1-r)*b.tc*b.pp.w())/result.pp.w();
//    result.wp = (a.wp*a.pp.w()*r+(1-r)*b.wp*b.pp.w())/result.pp.w();
    return result;
  }
  else
  {
    result.p = r*a.p+(1-r)*b.p;
    result.n = r*a.n+(1-r)*b.n;
    result.wp = r*a.wp+(1-r)*b.wp;
    result.tp = r*a.tp+(1-r)*b.tp;
    result.pp = r*a.pp+(1-r)*b.pp;

    result.tc=r*a.tc+(1-r)*b.tc;
    return result;
  }
}

/// Class Geometry
void Geometry::AddPoints(QVector<QVector3D> vs)
{
  for(int i=0; i<vs.size(); i++)
  {
    vecs.push_back(VertexInfo(vs[i]));
  }
}

double Geometry::Top()
{
  top = -1e20;
  for(int i=0; i<vecs.size(); i++)
  {
    if(vecs[i].pp.y()>top)
    {
      top=vecs[i].pp.y();
    }
  }
  return top;
}

double Geometry::Bottom()
{
  bottom = 1e20;
  for(int i=0; i<vecs.size(); i++)
  {
    if(vecs[i].pp.y()<bottom)
    {
      bottom=vecs[i].pp.y();
    }
  }
  return bottom;
}

void Geometry::SetNormal()
{
//    qDebug() << "vecs.size=" << vecs.size();
  QVector3D n = Normal();
  for(int i=0; i<vecs.size(); i++)
  {
    vecs[i].n = n;
  }
}

QVector3D Geometry::Normal()
{
  if(vecs.size()>=3)
  {
    QVector3D v0=vecs[0].p;
    QVector3D v1=vecs[1].p;
    QVector3D v2=vecs[2].p;
//      qDebug() << "v0=" << v0 << "v1=" << v1 << "v2=" << v2;
    return QVector3D::crossProduct(v2-v1, v0-v1).normalized();
  }
  return QVector3D(0, 0, 0);
}

bool Geometry::IsPlane()
{
  if(vecs.size()<3) return true;
  QVector3D n=Normal();
  for(int i=1; i<vecs.size(); i++)
  {
    QVector3D d = vecs[i].p-vecs[0].p;
    if(d.length()>1e-3 && fabs(QVector3D::dotProduct(d, n))>1e-3)
    {
      return false;
    }
  }
  return true;
}

bool Geometry::IsInside(QVector3D p)
{
  if(!IsPlane()) return false;
  QVector<QVector3D> vlist;
  for(int i=0; i<vecs.size(); i++)
  {
    vlist.push_back((vecs[i].p-p).normalized());
  }
  double sc = 1.0, ss = 0.0;
  for(int i=0; i<vlist.size(); i++)
  {
    int nn = (i+1)%vlist.size();
    double c = QVector3D::dotProduct(vlist[i], vlist[nn]);
    double s = QVector3D::crossProduct(vlist[i], vlist[nn]).length();
    double nc = sc*c-ss*s;
    double ns = ss*c+sc*s;
    sc = nc;
    ss = ns;
  }
  if(fabs(ss)>1e-2) return false;
  return true;
}

QVector4D Geometry::Plane()
{
  QVector4D n(Normal(), 0.0);
  n.setW(-QVector3D::dotProduct(Normal(), vecs[0].p));
  return n;
}

bool operator> (const Geometry& a, const Geometry& b) { return a.top> b.top; }
bool operator< (const Geometry& a, const Geometry& b) { return a.top< b.top; }
bool operator>=(const Geometry& a, const Geometry& b) { return a.top>=b.top; }
bool operator<=(const Geometry& a, const Geometry& b) { return a.top<=b.top; }
bool operator==(const Geometry& a, const Geometry& b) { return a.top==b.top; }

QDebug& operator<<(QDebug& s, const Geometry& g)
{
  s << "Geometry " << g.name << " @ " << g.vecs.size() << endl;
  for(int i=0; i<g.vecs.size(); i++)
  {
    s << "  vecs #" << i << ":" << " p:" << g.vecs[i].p << " pp:" << g.vecs[i].pp << " n:" << g.vecs[i].n << " tp:" << g.vecs[i].tp << " tc:" << g.vecs[i].tc << endl;
  }
  return s;
}
