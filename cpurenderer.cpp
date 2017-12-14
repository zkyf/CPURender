#include "cpurenderer.h"

QVector3D operator*(const QMatrix3x3& m, const QVector3D& x)
{
  return QVector3D(
      m.data()[0]*x.x()+m.data()[1]*x.y()+m.data()[2]*x.z(),
      m.data()[3]*x.x()+m.data()[4]*x.y()+m.data()[5]*x.z(),
      m.data()[6]*x.x()+m.data()[7]*x.y()+m.data()[8]*x.z()
      );
}

bool operator< (const ScanlinePoint& a, const ScanlinePoint& b) { return a.x()<b.x(); }

bool operator> (const DepthFragment& a, const DepthFragment& b) { return a.pos.z()> b.pos.z(); }
bool operator>=(const DepthFragment& a, const DepthFragment& b) { return a.pos.z()>=b.pos.z(); }
bool operator< (const DepthFragment& a, const DepthFragment& b) { return a.pos.z()< b.pos.z(); }
bool operator<=(const DepthFragment& a, const DepthFragment& b) { return a.pos.z()<=b.pos.z(); }
bool operator==(const DepthFragment& a, const DepthFragment& b) { return a.pos.z()==b.pos.z(); }

bool operator> (const Geometry& a, const Geometry& b) { return a.top> b.top; }
bool operator< (const Geometry& a, const Geometry& b) { return a.top< b.top; }
bool operator>=(const Geometry& a, const Geometry& b) { return a.top>=b.top; }
bool operator<=(const Geometry& a, const Geometry& b) { return a.top<=b.top; }
bool operator==(const Geometry& a, const Geometry& b) { return a.top==b.top; }

QDebug& operator<<(QDebug& s, const Geometry& g)
{
  s << "Geometry @ " << g.vecs.size() << endl;
  for(int i=0; i<g.vecs.size(); i++)
  {
    s << "  vecs #" << i << ":" << g.vecs[i] << endl;
  }
  return s;
}

QDebug& operator<<(QDebug& s, const ScanlinePoint& p)
{
  s << "sp " << "(" << p.x() << ", " << p.y() << ", " << p.z() << ") " << "hp=" << p.hp;
  return s;
}

QDebug& operator<<(QDebug& s, const Scanline& line)
{
  s << "Scanline @" << line.size() << endl;
  for(int i=0; i<line.size(); i++)
  {
    s << "  hp #" << i << ":" << line[i] << endl;
  }
  return s;
}

QDebug& operator<<(QDebug& s, const ColorPixel& p)
{
  s << "(" << p.r << ", " << p.g << ", " << p.b << ", " << p.a << ")";
  return s;
}

CPURenderer::CPURenderer(QSize s) :
  size(s),
  colorBuffer(s),
  depthBuffer(s),
  stencilBuffer(nullptr)
{
  int bufferCount = size.width()*size.height();
  stencilBuffer = new uchar[bufferCount];
  memset(stencilBuffer, 0, sizeof(uchar)*bufferCount);

  projection.perspective(60, 1.0, 0.1, 100.0);
  nearPlane=0.1;
  farPlane = 100.0;
}

CPURenderer::~CPURenderer()
{
  delete[] stencilBuffer;
}

uchar* CPURenderer::Render()
{
  qDebug() << "CPURenderer Start";
  // vertex shader
  {
    qDebug() << "vertex shader";
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      for(int vi=0; vi<i->vecs.size(); vi++)
      {
        VertexShader(i->vecs[vi], i->norms[vi], i->texcoords[vi]);
      }
    }

    for(int i=0; i<geos.size(); i++)
    {
      qDebug() << "after: geos#" << i;
      for(int j=0; j<geos[i].vecs.size(); j++)
      {
        qDebug() << "  vecs#" << j << ":" << geos[i].vecs[j];
      }
    }

    qDebug() << "vertex shader ended";
  }

  // geometry shader
//  {
//    qDebug() << "geometry shader";
//    int geosNum = geos.size();
//    int count = 0;
//    for(GI i= geos.begin(); count<geosNum; i++)
//    {
//      GeometryShader(*i);
//    }
//    qDebug() << "geometry shader ended";
//  }

  // post process & clipping
  {
    qDebug() << "post process & clipping";
    qDebug() << "farPlane=" << farPlane << ", nearPlane=" << nearPlane;
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      qDebug() << "now processing: " << *i;
      Geometry ng;
      ng.ambient=i->ambient;
      ng.diffuse=i->diffuse;
      ng.specular=i->specular;
      bool dirt = false;
      bool last = false; // true=in, false=out

      QVector3D p = i->vecs[0];
      qDebug() << "#0 @" << p;
      if(p.x()>=-1 && p.y()<=1 && p.z()<= farPlane && p.z()>=nearPlane)
      {
        ng.vecs.push_back(p);
        ng.norms.push_back(i->norms[0]);
        ng.texcoords.push_back(i->texcoords[0]);
        last = true;
      }
      else
      {
        dirt=true;
        last = false;
      }
      qDebug() << "# 0 :" << dirt << last;

      for(int j=1; j<i->vecs.size(); j++)
      {
        QVector3D p = i->vecs[j];
        if(p.x()>=-1 && p.y()<=1 && p.z()<= farPlane && p.z()>=nearPlane)
        {
          qDebug() << "this==true";
          ng.vecs.push_back(p);
          ng.norms.push_back(i->norms[j]);
          ng.texcoords.push_back(i->texcoords[j]);

          if(last==false)
          {
            qDebug() << "last==false";
            QVector3D l=i->vecs[j-1];
            QVector3D t=p;

            if(t.x()<-1)
            {
              t.setY(fabs((-1-t.x()))/fabs((l.x()-t.x()))*l.y()+fabs((l.x()+1))/fabs((l.x()-t.x()))*t.y());
              t.setZ(fabs((-1-t.x()))/fabs((l.x()-t.x()))*l.z()+fabs((l.x()+1))/fabs((l.x()-t.x()))*t.z());
              t.setX(-1);
            }
            if(t.x()>1)
            {
              t.setY(fabs((1-t.x()))/fabs((l.x()-t.x()))*l.y()+fabs((l.x()-1))/fabs((l.x()-t.x()))*t.y());
              t.setZ(fabs((1-t.x()))/fabs((l.x()-t.x()))*l.z()+fabs((l.x()-1))/fabs((l.x()-t.x()))*t.z());
              t.setX(1);
            }

            if(t.y()<-1)
            {
              t.setX(fabs((-1-t.y()))/fabs((l.y()-t.y()))*l.x()+fabs((l.y()+1))/fabs((l.y()-t.y()))*t.x());
              t.setZ(fabs((-1-t.y()))/fabs((l.y()-t.y()))*l.z()+fabs((l.y()+1))/fabs((l.y()-t.y()))*t.z());
              t.setY(-1);
            }
            if(t.y()>1)
            {
              t.setX(fabs((1-t.y()))/fabs((l.y()-t.y()))*l.x()+fabs((l.y()-1))/fabs((l.y()-t.y()))*t.x());
              t.setZ(fabs((1-t.y()))/fabs((l.y()-t.y()))*l.z()+fabs((l.y()-1))/fabs((l.y()-t.y()))*t.z());
              t.setY(1);
            }

            if(t.z()<nearPlane)
            {
              t.setX(fabs((nearPlane-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-nearPlane))/fabs((l.z()-t.z()))*t.x());
              t.setY(fabs((nearPlane-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-nearPlane))/fabs((l.z()-t.z()))*t.y());
              t.setZ(nearPlane);
            }
            if(t.z()>farPlane)
            {
              t.setX(fabs((farPlane-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-farPlane))/fabs((l.z()-t.z()))*t.x());
              t.setY(fabs((farPlane-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-farPlane))/fabs((l.z()-t.z()))*t.y());
              t.setZ(farPlane);
            }

            qDebug() << "result t=" << t;
            ng.vecs.push_back(t);
          }

          ng.vecs.push_back(p);
          last=true;
        }
        else
        {
          qDebug() << "this==false";
          dirt=true;
          if(last==true)
          {
            qDebug() << "last==true";
            QVector3D l=i->vecs[j-1];
            QVector3D t=p;

            if(t.x()<-1)
            {
              t.setY(fabs((-1-t.x()))/fabs((l.x()-t.x()))*l.y()+fabs((l.x()+1))/fabs((l.x()-t.x()))*t.y());
              t.setZ(fabs((-1-t.x()))/fabs((l.x()-t.x()))*l.z()+fabs((l.x()+1))/fabs((l.x()-t.x()))*t.z());
              t.setX(-1);
            }
            if(t.x()>1)
            {
              t.setY(fabs((1-t.x()))/fabs((l.x()-t.x()))*l.y()+fabs((l.x()-1))/fabs((l.x()-t.x()))*t.y());
              t.setZ(fabs((1-t.x()))/fabs((l.x()-t.x()))*l.z()+fabs((l.x()-1))/fabs((l.x()-t.x()))*t.z());
              t.setX(1);
            }

            if(t.y()<-1)
            {
              t.setX(fabs((-1-t.y()))/fabs((l.y()-t.y()))*l.x()+fabs((l.y()+1))/fabs((l.y()-t.y()))*t.x());
              t.setZ(fabs((-1-t.y()))/fabs((l.y()-t.y()))*l.z()+fabs((l.y()+1))/fabs((l.y()-t.y()))*t.z());
              t.setY(-1);
            }
            if(t.y()>1)
            {
              t.setX(fabs((1-t.y()))/fabs((l.y()-t.y()))*l.x()+fabs((l.y()-1))/fabs((l.y()-t.y()))*t.x());
              t.setZ(fabs((1-t.y()))/fabs((l.y()-t.y()))*l.z()+fabs((l.y()-1))/fabs((l.y()-t.y()))*t.z());
              t.setY(1);
            }

            if(t.z()<nearPlane)
            {
              t.setX(fabs((nearPlane-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-nearPlane))/fabs((l.z()-t.z()))*t.x());
              t.setY(fabs((nearPlane-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-nearPlane))/fabs((l.z()-t.z()))*t.y());
              t.setZ(nearPlane);
            }
            if(t.z()>farPlane)
            {
              t.setX(fabs((farPlane-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-farPlane))/fabs((l.z()-t.z()))*t.x());
              t.setY(fabs((farPlane-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-farPlane))/fabs((l.z()-t.z()))*t.y());
              t.setZ(farPlane);
            }

            ng.vecs.push_back(t);
          }
          last=false;
        }
      }

      if(dirt)
      {
        *i=ng;
      }
    }

    for(int i=0; i<geos.size(); i++)
    {
      qDebug() << "after: geos#" << i;
      for(int j=0; j<geos[i].vecs.size(); j++)
      {
        qDebug() << "  vecs#" << j << ":" << geos[i].vecs[j];
      }
    }
    qDebug() << "post process & clipping ended";
  }

  // rasterization
  {
    qDebug() << "rasterization";

    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      i->Top(); i->Bottom();
    }
    qSort(geos);

    qDebug() << "geos sorted";

    float top=-1e20, bottom=1e20;
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      qDebug() << *i;
      if(i->vecs.size()<3)
      {
        continue;
      }

      if(i->top>top) top=i->top;
      if(i->bottom<bottom) bottom=i->bottom;
    }
    int tt=ToScreenY(top);
    int bb=ToScreenY(bottom);
    qDebug() << "tt=" << tt << " bb=" << bb;

    for(int yy=tt; yy<bb; yy++)
    {
      float y=ToProjY(yy);
      // generate a scanline
      Scanline s;
      qDebug() << "generate scanline";
      for(GI i=geos.begin(); i!=geos.end(); i++)
      {
//        qDebug() << "  for geo@" << *i;
        qDebug() << "geo.top=" << ToScreenY(i->top) << " ," << "geo.bottom=" << ToScreenY(i->bottom);
        if(ToScreenY(i->top)>yy || ToScreenY(i->bottom)<yy)
        {
          continue;
        }

        int c=0;
        qDebug() << "finding hp";
        for(int j=0; j<i->vecs.size(); j++)
        {
          int next=(j+1)%i->vecs.size();
          qDebug() << "cond=" << (y-i->vecs[j].y())*(i->vecs[next].y()-y);
          if((y-i->vecs[j].y())*(i->vecs[next].y()-y)>=0)
          {
            c++;
            float r=(i->vecs[j].y()-y)/(i->vecs[j].y()-i->vecs[next].y());
            ScanlinePoint np(r*i->vecs[next]+(1-r)*i->vecs[j]);

            np.geo=i;
            np.hp=c;
            s.push_back(np);
          }
        }
      }

      qSort(s);

      qDebug() << "sl @yy=" << yy << " y=" << y << ":" << s;

      // fill depth values
      qDebug() << "fill depth values";
      QSet<GI> nowPoly;
      for(int pid=0; pid<s.size(); pid++)
      {
        // fill depth buffer here;

        qDebug() << "hp #" << pid;

        if(s[pid].hp%2==1)
        {
          nowPoly.insert(s[pid].geo);
        }
        else
        {
          nowPoly.remove(s[pid].geo);
        }

        float nn;
        if(pid<s.size()-1)
        {
          nn=s[pid].x();
        }
        else
        {
          nn=1.0;
        }

        float nowz=s[pid].z();
        for(int xx=ToScreenX(s[pid].x()); ToProjX(xx)<nn; xx++)
        {
          for(QSet<GI>::iterator geoToDraw = nowPoly.begin(); geoToDraw!=nowPoly.end(); geoToDraw++)
          {
            qDebug() << "depthfragment pushed @" << xx << ", " << yy;
            DepthFragment ng;
            ng.pos.setX(ToProjX(xx));
            ng.pos.setY(y);
            ng.pos.setZ(nowz);
            ng.geo=*geoToDraw;
            nowz+=ng.geo->dz;
            // calculate deoth value of a fragment here
            depthBuffer.buffer[xx+yy*size.width()].chain.push_back(ng);
          }
        }
      }
    }

    qDebug() << "rasterization ended";
  }

  // fragment shader
  {
    qDebug() << "fragment shader";
    for(int i=0; i<size.width()*size.height(); i++)
    {
      for(int j=0; j<depthBuffer.buffer[i].chain.size(); j++)
      {
        FragmentShader(depthBuffer.buffer[i].chain[j]);
//        qDebug() << "color @" << i << "=" << depthBuffer.buffer[i].chain[j].color;
      }
    }
    qDebug() << "fragment shader ended";
  }

  // per sample operations
  {
    qDebug() << "per sample operation";
    for(int i=0; i<size.width()*size.height(); i++)
    {
      qSort(depthBuffer.buffer[i].chain);
      colorBuffer.buffer[i] = ColorPixel(0.0, 0.0, 0.0, 0.0);
      for(int j=0; j<depthBuffer.buffer[i].chain.size(); j++)
      {
        colorBuffer.buffer[i] = colorBuffer.buffer[i]+depthBuffer.buffer[i].chain[j].color;
//        qDebug() << "pixel @ (" << i%size.width() << ", " << i/size.height() << ")" << "=" << colorBuffer.buffer[i];
        if(colorBuffer.buffer[i].a >=0.99) // not transparent
        {
          break;
        }
      }
    }
    qDebug() << "per sample operation ended";
  }

  return colorBuffer.ToUcharArray();
}

void CPURenderer::VertexShader(QVector3D &p, QVector3D &n, QVector2D &tc)
{
  QVector4D pp(p, 1.0);
  pp = projection * camera.toMatrix() * transform.toMatrix() * pp;
  p.setX(pp.x());
  p.setY(pp.y());
  p.setZ(pp.z());

  n = transform.rotation().toRotationMatrix()*n;
}

void CPURenderer::GeometryShader(Geometry &geo)
{

}

void CPURenderer::FragmentShader(DepthFragment &frag)
{
  frag.color = ColorPixel(1.0, 1.0, 1.0, 1.0);
}

void CPURenderer::AddGeometry(const Geometry &geo)
{
  geos.push_back(geo);
}

int CPURenderer::ToScreenX(float x)
{
  return (int)((x+1.0)/2*size.width());
}

int CPURenderer::ToScreenY(float y)
{
  return (int)((-y+1.0)/2*size.height());
}

float CPURenderer::ToProjX(int x)
{
  return x*1.0/size.width()*2-1.0;
}

float CPURenderer::ToProjY(int y)
{
  return -(y*1.0/size.height()*2-1.0);
}

void CPURenderer::Resize(QSize w)
{
  size=w;
}

void CPURenderer::WorldTranslate(QVector3D trans)
{
  transform.translate(trans);
}

void CPURenderer::WorldRotate(QVector3D axis, double angle)
{
  transform.rotate(angle, axis);
}

QVector3D CPURenderer::WorldTranslation()
{
  return transform.translation();
}

QQuaternion CPURenderer::WorldRotation()
{
  return transform.rotation();
}

void CPURenderer::CamTranslate(QVector3D trans)
{
  camera.translate(trans);
}

void CPURenderer::CamRotate(QVector3D axis, double angle)
{
  camera.rotate(angle, axis);
}

QVector3D CPURenderer::CamTranslation()
{
  return camera.translation();
}

QQuaternion CPURenderer::CamRotation()
{
  return camera.rotation();
}

QSize CPURenderer::Size()
{
  return size;
}

void CPURenderer::AddLight(Light light)
{
  lights.push_back(light);
}
