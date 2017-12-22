#include "cpurenderer.h"

QVector3D operator*(const QMatrix3x3& m, const QVector3D& x)
{
  return QVector3D(
      m.data()[0]*x.x()+m.data()[1]*x.y()+m.data()[2]*x.z(),
      m.data()[3]*x.x()+m.data()[4]*x.y()+m.data()[5]*x.z(),
      m.data()[6]*x.x()+m.data()[7]*x.y()+m.data()[8]*x.z()
      );
}

bool operator< (const ScanlinePoint& a, const ScanlinePoint& b)
{
  if(a.geo!=b.geo)
    return a.geo<b.geo;
  else
    return a.x()<b.x();
}

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

bool operator< (const EdgeListItem& a, const EdgeListItem& b)
{
  if(a.geo->vecs[a.tid].pp.y()!=b.geo->vecs[b.tid].pp.y())
    return a.geo->vecs[a.tid].pp.y()<b.geo->vecs[b.tid].pp.y();
  else
    return a.geo->vecs[a.bid].pp.y()<b.geo->vecs[b.bid].pp.y();
}

QDebug& operator<<(QDebug& s, const Geometry& g)
{
  s << "Geometry " << g.name << " @ " << g.vecs.size() << endl;
  for(int i=0; i<g.vecs.size(); i++)
  {
    s << "  vecs #" << i << ":" << " p:" << g.vecs[i].p << " pp:" << g.vecs[i].pp << " n:" << g.vecs[i].n << " tp:" << g.vecs[i].tp << " tc:" << g.vecs[i].tc << endl;
  }
  return s;
}

QDebug& operator<<(QDebug& s, const ScanlinePoint& p)
{
  s << "sp " << "(" << p.x() << ", " << p.y() << ", " << p.z() << ") ";
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

QDebug& operator<<(QDebug& s, const DepthFragment& f)
{
  s << f.geo->name << " color=" << f.color << " pos=" << f.pos << " normal=" << f.v.n << " tc=" << f.v.tc << endl;
  s << *f.geo;
  return s;
}

QDebug& operator<<(QDebug& s, const DepthPixel& d)
{
  for(int i=0; i<d.chain.size(); i++)
  {
    s << "chain #" << i << ":" << d.chain[i] << endl;
  }
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

//  projection.lookAt(camera.translation(), camera.translation()+camera.forward(), camera.up());
  ////qDebug() << "projection" << projection;
  projection.perspective(60, 1.0, 0.1, 100.0);
//  projection.ortho(-1, 1, -1, 1, 0, 100);
  ////qDebug() << "projection" << projection;
  nearPlane=0.1;
  farPlane = 100.0;
}

CPURenderer::~CPURenderer()
{
  delete[] stencilBuffer;
}

uchar* CPURenderer::Render()
{
  float epsy = 0.1/size.height();
  float epsx = 0.1/size.width();
  colorBuffer.Clear();
  depthBuffer.Clear();

  ////qDebug() << "CPURenderer Start";
  // vertex shader
  {
    geos = input;
    //qDebug() << "vertex shader";
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      //qDebug() << "before:" << *i;
      for(int vi=0; vi<i->vecs.size(); vi++)
      {
        i->vecs[vi]=VertexShader(i->vecs[vi]);
      }

      //qDebug() << "after:" << *i;
    }
    //qDebug() << "vertex shader ended";
  }

  // pre-clipping
  {
    //qDebug() << "pre-clipping";
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      //qDebug() << "pre-clipping z<0";
      QVector<QVector4D> vecs;
      for(int v=0; v<i->vecs.size(); v++)
      {
        vecs.push_back(i->vecs[v].tp);
      }
      bool dirty;
      Clip(QVector4D(0, 0, 1, 0), false, vecs, i, dirty);
      //qDebug() << "after: " << *i;

      if(dirty && i->vecs.size()>0)
      {
        for(int vi=0; vi<i->vecs.size(); vi++)
        {
          i->vecs[vi]=VertexShader(i->vecs[vi]);
        }
      }

      //qDebug() << "re-vertex-shader after:" << *i;
    }
    //qDebug() << "pre-clipping ended";
  }

  // geometry shader
//  {
//    ////qDebug() << "geometry shader";
//    int geosNum = geos.size();
//    int count = 0;
//    for(GI i= geos.begin(); count<geosNum; i++)
//    {
//      GeometryShader(*i);
//    }
//    ////qDebug() << "geometry shader ended";
//  }

  // new clipping
  {
    //qDebug() << "new clipping";
    bool dirty;
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      //qDebug() << "before: " << *i;

      {
        //qDebug() << "x>-1";
        QVector<QVector4D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(1, 0, 0, 1), true, vecs, i, dirty);
      }

      {
        //qDebug() << "x<1";
        QVector<QVector4D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(-1, 0, 0, 1), true, vecs, i, dirty);
      }

      {
        //qDebug() << "y>-1";
        QVector<QVector4D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, 1, 0, 1), true, vecs, i, dirty);
      }

      {
        //qDebug() << "y<1";
        QVector<QVector4D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, -1, 0, 1), true, vecs, i, dirty);
      }

      {
        //qDebug() << "z>-1";
        QVector<QVector4D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, 0, 1, 1), true, vecs, i, dirty);
      }

      {
        //qDebug() << "z<1";
        QVector<QVector4D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, 0, -1, 1), true, vecs, i, dirty);
      }

      //qDebug() << "after: " << *i;
    }
  }

  // calculate dz and dy
  //qDebug() << "calculate dz and dy";
  for(GI i=geos.begin(); i!=geos.end(); i++)
  {
    if(i->vecs.size()<3) continue;
    QVector4D v0=i->vecs[0].pp;
    QVector4D v1=i->vecs[1].pp;
    QVector4D v2=i->vecs[2].pp;
    Mat A(2, 2, CV_32F, Scalar::all(0));
    A.at<float>(0, 0)=v2.x()-v0.x();
    A.at<float>(0, 1)=v2.y()-v0.y();
    A.at<float>(1, 0)=v2.x()-v1.x();
    A.at<float>(1, 1)=v2.y()-v1.y();
    Mat b(2, 1, CV_32F, Scalar::all(0));
    b.at<float>(0, 0)=v2.z()-v0.z();
    b.at<float>(1, 0)=v2.z()-v1.z();
    Mat dxy=A.inv()*b;

    i->dz = dxy.at<float>(0, 0)/size.width()*2;
    i->dzy = dxy.at<float>(1, 0)/size.height()*2;
    ////qDebug() << "geo " << *i;
    ////qDebug() << "dz=" << i->dz;
  }
  //qDebug() << "calculate dz and dy ended";

  // generate edge list
  //qDebug() << "generate edge list";
  QVector<EdgeListItem> edgeList;
  int nowe;
  {
    //qDebug() << "regional scanline";
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      for(int v=0; v<i->vecs.size(); v++)
      {
        EdgeListItem item;
        item.geo=i;
        int next=(v+1)%i->vecs.size();
        if(i->vecs[v].pp.y()>i->vecs[next].pp.y())
        {
          item.tid=v; item.bid=next;
        }
        else
        {
          item.tid=next; item.bid=v;
        }
        edgeList.push_back(item);
      }
    }

    qSort(edgeList);
    nowe=edgeList.size()-1;
  }

  //qDebug() << "generated edge list in total " << edgeList.size();
  for(int i=edgeList.size()-1; i>=0; i--)
  {
    //qDebug() << " edgelist #" << i << " : " << edgeList[i].geo->name << " tid=" << edgeList[i].tid << "=" << edgeList[i].geo->vecs[edgeList[i].tid].pp << " bid=" << edgeList[i].bid << "=" << edgeList[i].geo->vecs[edgeList[i].bid].pp;
  }

  // new rasterization
  //qDebug() << "new rasterization";
  if(edgeList.size()>0)
  {
    while(nowe>=0)
    {
      GI geo = edgeList[nowe].geo;
      if(geo->vecs[edgeList[nowe].tid].pp.y()>1.0 && geo->vecs[edgeList[nowe].bid].pp.y()>1.0)
      {
        nowe--;
      }
      else
      {
        break;
      }
    }

    int topyy=ToScreenY(edgeList[nowe].geo->vecs[edgeList[nowe].tid].pp.y());
    if(topyy<0) topyy=0;
    float topy=ToProjY(topyy);

    Scanline scanline;
    //qDebug() << "topy=";
    while(nowe>=0)
    {
      // find points for initial scanline

      //qDebug() << "nowe=" << nowe << " on " << edgeList[nowe].geo->name << " tid=" << edgeList[nowe].tid << " bid=" << edgeList[nowe].bid;
      GI geo=edgeList[nowe].geo;
      int tid=edgeList[nowe].tid;
      int bid=edgeList[nowe].bid;
      if(ToScreenY(geo->vecs[tid].pp.y())==ToScreenY(geo->vecs[bid].pp.y()))
      {
        //qDebug() << "ignore horizontal edge";
        nowe--;
        continue;
      }
      //qDebug() << "geo->vecs[tid].pp.y()-topy=" << geo->vecs[tid].pp.y()-topy << "geo->vecs[bid].pp.y()-nexty=" << geo->vecs[bid].pp.y()-topy;
      if(geo->vecs[tid].pp.y()>=topy && geo->vecs[bid].pp.y()<=topy)
      {
        float r=(topy-geo->vecs[bid].pp.y())/(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y());

        ScanlinePoint np(QVector3D(r*geo->vecs[tid].pp+(1-r)*geo->vecs[bid].pp));
        np.v = VertexInfo::Intersect(geo->vecs[tid], geo->vecs[bid], r);
        np.geo=geo;
        np.e = nowe;
        np.dx = (geo->vecs[tid].pp.x()-geo->vecs[bid].pp.x())/(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y())/size.height()*2;

        scanline.push_back(np);
        nowe--;
      }
      else
      {
        break;
      }
    }
    qSort(scanline);

    for(int yy=topyy; yy<size.height(); yy++)
    {
      bool error=false;
      float y=ToProjY(yy);
      for(GI i=geos.begin(); i!=geos.end(); i++)
      {
        i->inout=false;
      }

      //qDebug() << yy << " : scanline.size()=" << scanline.size();
      for(int pid=0; pid<scanline.size(); pid++)
      {
        //qDebug() << "pid #" << pid << " p=" << scanline[pid] << " of geometry:" << scanline[pid].geo->name << edgeList[scanline[pid].e].tid << edgeList[scanline[pid].e].bid << scanline[pid].dx*size.height()/2;
      }

      for(int pid=0; pid<scanline.size(); pid++)
      {
        if(pid>0 && pid<scanline.size()-1 && (scanline[pid-1].geo==scanline[pid].geo && scanline[pid+1].geo==scanline[pid].geo) && (fabs(scanline[pid-1].x()-scanline[pid].x())<epsx || fabs(scanline[pid+1].x()-scanline[pid].x())<epsx))
        {
          //qDebug() << "pid=" << pid << " skipped";
          continue;
        }

        ScanlinePoint& p=scanline[pid];
        p.geo->inout=!p.geo->inout;

        if(p.geo->inout)
        {
          if(pid==scanline.size()-1 || p.geo!=scanline[pid+1].geo)
          {
            //qDebug() << "Error: Odd number of scan points @ yy=" << yy << " p=" << p << " of geometry:";
            error=true;
            //qDebug() << *p.geo;
          }

          int nextxx;
          if(pid<scanline.size()-1) nextxx=ToScreenX(scanline[pid+1].x());
          else nextxx=size.width();
          if(error) nextxx=ToScreenX(scanline[pid].x())+1;
          if(nextxx>=size.width()) nextxx=size.width();
          float nowz=p.z();
          for(int xx=ToScreenX(p.x()); xx<nextxx; xx++)
          {
            //          ////qDebug() << "xx=" << xx;
            DepthFragment ng;
            ng.pos.setX(ToProjX(xx));
            ng.pos.setY(y);
            ng.index=xx+yy*size.width();
            float r=(ng.pos.x()-scanline[pid].x())/(scanline[pid+1].x()-scanline[pid].x());
            //          if(yy=size.height()/2)
            //          {
            //            ////qDebug() << "now geo=" << s[pid].geo->name << "xx=" << xx << "nowz=" << nowz << s[pid+1].z() << s[pid].z() << s[pid+1].x() << s[pid].x();
            //          }
            ng.pos.setZ(nowz);
            ng.geo=scanline[pid].geo;
            ng.v = VertexInfo::Intersect(scanline[pid+1].v, scanline[pid].v, r);
            nowz+=ng.geo->dz;
            // calculate deoth value of a fragment here
            depthBuffer.buffer[xx+yy*size.width()].chain.push_back(ng);
          }
        }
        else
        {
          if(pid<scanline.size()-1 && (pid==scanline.size()-2 || (scanline[pid+1].geo==scanline[pid].geo && scanline[pid+2].geo!=scanline[pid+1].geo)))
          {
            scanline[pid].geo->inout=!scanline[pid].geo->inout;
          }
        }
      }

      // generate new scanline
      {
        float nexty=ToProjY(yy+1);
        for(int pid=0; pid<scanline.size(); pid++)
        {
          ScanlinePoint& p=scanline[pid];
          GI geo=p.geo;
          EdgeListItem& e=edgeList[p.e];
          if(geo->vecs[e.bid].pp.y()>nexty)
          {
//            if(error)
            {
              //qDebug() << "pid #" << pid << " removed";
            }
            scanline.remove(pid);
            pid--;
            continue;
          }
          else
          {
            p.setX(p.x()-p.dx);
            p.setY(p.y()-2.0/size.height());
            p.setZ(p.z()-p.dx*geo->dz/(2.0/size.width())-geo->dzy);

            float r = (p.y()-geo->vecs[e.bid].pp.y()) / (geo->vecs[e.tid].pp.y()-geo->vecs[e.bid].pp.y());
            p.v = VertexInfo::Intersect(geo->vecs[e.tid], geo->vecs[e.bid], r);
          }
        }

        while(nowe>=0)
        {
//          if(error)
          {
            //qDebug() << "checking new e #" << nowe << " :" << edgeList[nowe].geo->name << edgeList[nowe].tid << edgeList[nowe].bid;
          }
          GI geo=edgeList[nowe].geo;
          int tid=edgeList[nowe].tid;
          int bid=edgeList[nowe].bid;
          //qDebug() << "horizontal? fabs(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y())=" << fabs(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y()) << "epsy=" << epsy;
          //qDebug() << "ToScreenY(geo->vecs[tid].pp.y())=" << ToScreenY(geo->vecs[tid].pp.y()) << "ToScreenY(geo->vecs[bid].pp.y())=" << ToScreenY(geo->vecs[bid].pp.y());
          if(ToScreenY(geo->vecs[tid].pp.y())==ToScreenY(geo->vecs[bid].pp.y()))
          {
//            if(error)
            {
              //qDebug() << "ignore horizontal edge";
            }
            nowe--;
            continue;
          }
//          if(error)
          {
            //qDebug() << "geo->vecs[tid].pp.y()-nexty=" << geo->vecs[tid].pp.y()-nexty << "geo->vecs[bid].pp.y()-nexty=" << geo->vecs[bid].pp.y()-nexty;
          }
          if(geo->vecs[tid].pp.y()>=nexty && geo->vecs[bid].pp.y()<=nexty)
          {
            float r=(nexty-geo->vecs[bid].pp.y())/(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y());

            ScanlinePoint np(QVector3D(r*geo->vecs[tid].pp+(1-r)*geo->vecs[bid].pp));
            np.geo=geo;
            np.e = nowe;
            np.v = VertexInfo::Intersect(geo->vecs[tid], geo->vecs[bid], r);
            np.dx = (geo->vecs[tid].pp.x()-geo->vecs[bid].pp.x())/(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y())/size.height()*2;

            scanline.push_back(np);
            if(error)
            {
              //qDebug() << "edge added";
            }
            nowe--;
          }
          else
          {
            break;
          }
        }

        qSort(scanline);
      }
    }
  }
  //qDebug() << "new rasterization ended";

  // fragment shader
  {
    //qDebug() << "fragment shader";
    for(int i=0; i<size.width()*size.height(); i++)
    {
      for(int j=0; j<depthBuffer.buffer[i].chain.size(); j++)
      {
        FragmentShader(depthBuffer.buffer[i].chain[j]);
//        ////qDebug() << "color @" << i << "=" << depthBuffer.buffer[i].chain[j].color;
      }
    }
    //qDebug() << "fragment shader ended";
  }

  // per sample operations
  {
    //qDebug() << "per sample operation";
    for(int i=0; i<size.width()*size.height(); i++)
    {
      qSort(depthBuffer.buffer[i].chain);
      colorBuffer.buffer[i] = ColorPixel(1.0, 1.0, 1.0, 0.0);
      for(int j=0; j<depthBuffer.buffer[i].chain.size(); j++)
      {
        colorBuffer.buffer[i] = colorBuffer.buffer[i]+depthBuffer.buffer[i].chain[j].color;
        if(i/size.width()==size.height()/2)
        {
//          //qDebug() << "depthBuffer @ (" << i%size.width() << ", " << i/size.height() << ")[" << j << "]=" << depthBuffer.buffer[i].chain[j].color;
        }
        if(colorBuffer.buffer[i].a >=0.99) // not transparent
        {
          break;
        }
      }
//      colorBuffer.buffer[i] = colorBuffer.buffer[i]+ColorPixel(1.0, 1.0, 1.0, 0.0);
//      //qDebug() << "colorBuffer @ (" << i%size.width() << ", " << i/size.height() << ")=" << colorBuffer.buffer[i] << depthBuffer.buffer[i].chain.size();
    }

    ////qDebug() << "center pixel:";
    int xc = size.width()/2+10;
    int yc = size.height()/2;
    for(int j=0; j<depthBuffer.buffer[xc+yc*size.width()].chain.size(); j++)
    {
      ////qDebug() << "frag #" << j << ":" << depthBuffer.buffer[xc+yc*size.width()].chain[j].pos << depthBuffer.buffer[xc+yc*size.width()].chain[j].color;
    }

    //qDebug() << "per sample operation ended";
  }

  uchar* result = colorBuffer.ToUcharArray();

  return result;
}

DepthPixel CPURenderer::DepthPixelAt(int x, int y)
{
  return depthBuffer.buffer[x+y*size.width()];
}

void CPURenderer::GeometryShader(Geometry &geo)
{

}

void CPURenderer::AddGeometry(const Geometry &geo)
{
  input.push_back(geo);
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
  depthBuffer.Resize(w);
  colorBuffer.Resize(w);

  projection.setToIdentity();
//  projection.lookAt(camera.translation(), camera.translation()+camera.forward(), camera.up());
  projection.perspective(60, 1.0*w.width()/w.height(), 0.1, 100.0);
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

void CPURenderer::Clip(QVector4D A, bool dir, QVector<QVector4D> g, GI i, bool &dirty)
{
  if(g.size()<3) return;
  if(g.size()!=i->vecs.size()) return;

  for(int vid=0; vid<g.size(); vid++)
  {
    g[vid].setW(1.0);
  }

  bool lastStatus = (QVector4D::dotProduct(A, g.last())<0) ^ dir;
  QVector<float> result;
  for(int i=0; i<g.size(); i++)
  {
    if((QVector4D::dotProduct(A, g[i])<0) ^ dir)
    {
      //qDebug() << "p #" << i << g[i] << "on " << dir << "side of " << A;
      if(!lastStatus)
      {
        QVector4D t(-g[i]+g[(i+g.size()-1)%g.size()]);
        t.setW(0);
        QVector4D p0(g[i]);
        float k=QVector4D::dotProduct(-A, p0)/QVector4D::dotProduct(A, t);
        //qDebug() << "out-in point:" << p0+k*t << QVector4D::dotProduct(p0+k*t, A);
        result.push_back(k);
      }
      else
      {
        result.push_back(10.0);
      }
    }
    else
    {
      //qDebug() << "p #" << i << g[i] << "on " << !dir << "side of " << A;
      if(lastStatus)
      {
        QVector4D t(-g[i]+g[(i+g.size()-1)%g.size()]);
        t.setW(0);
        QVector4D p0(g[i]);
        float k=QVector4D::dotProduct(-A, p0)/QVector4D::dotProduct(A, t);
        //qDebug() << "in-out point:" << p0+k*t << QVector4D::dotProduct(p0+k*t, A);
        result.push_back(-k);
      }
      else
      {
        result.push_back(-10);
      }
    }

    lastStatus=(QVector4D::dotProduct(A, g[i])<0) ^ dir;
  }

  {
    QVector<VertexInfo> ng;
    dirty=false;
    for(int v=0; v<i->vecs.size(); v++)
    {
      //qDebug() << "result #" << v << "=" << result[v];
      if(result[v]>5.0)
      {
        ng.push_back(i->vecs[v]);
      }
      else if(result[v]>0)
      {
        VertexInfo nf;
        nf = VertexInfo::Intersect(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()], i->vecs[v], result[v]);
        ng.push_back(nf);

        ng.push_back(i->vecs[v]);
        dirty=true;
      }
      else if(result[v]>-1)
      {
        VertexInfo nf;
        nf = VertexInfo::Intersect(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()], i->vecs[v], -result[v]);
        ng.push_back(nf);
        dirty=true;
      }
      else
      {
        dirty=true;
      }
    }
    if(dirty) i->vecs=ng;
  }
}

int CPURenderer::AddTexture(QImage text)
{
  textures.push_back(text);
  return textures.size()-1;
}

ColorPixel CPURenderer::Sample(int tid, float u, float v, bool bi)
{
  u=u+(int)u; if(u<0) u+=1.0; u-=(int)u;
  v=v+(int)v; if(v<0) v+=1.0; v-=(int)v;

  int w=textures[tid].width();
  int h=textures[tid].height();
  int ll=u*(w-1);
  ll = (ll % w + w) % w;
  int tt=(1-v)*(h-1);
  tt = (tt % h + h) % h;
  QColor lt = textures[tid].pixelColor(ll, tt);
  if(!bi)
  {
    return ColorPixel(lt.redF(),  lt.greenF(), lt.blueF(), lt.alphaF());
  }
  else
  {
    QColor rt = textures[tid].pixelColor((ll+1)%w, tt);
    QColor lb = textures[tid].pixelColor(ll, (tt+1)%h);
    QColor rb = textures[tid].pixelColor((ll+1)%w, (tt+1)%h);

    float uu=u*(w-1);
    float vv=(1-v)*(h-1);

    float ur=uu-ll;
    float vr=vv-tt;

    float ltr = (1-ur)*(1-vr);
    float rtr = ur*(1-vr);
    float lbr = (1-ur)*vr;
    float rbr = ur*vr;

    return ColorPixel(
          lt.redF()*ltr+rt.redF()*rtr+lb.redF()*lbr+rb.redF()*rbr,
          lt.greenF()*ltr+rt.greenF()*rtr+lb.greenF()*lbr+rb.greenF()*rbr,
          lt.blueF()*ltr+rt.blueF()*rtr+lb.blueF()*lbr+rb.blueF()*rbr,
          lt.alphaF()*ltr+rt.alphaF()*rtr+lb.alphaF()*lbr+rb.alphaF()*rbr
          );
  }
}

VertexInfo CPURenderer::VertexShader(VertexInfo v)
{
  QVector4D pp(v.p, 1.0);
  pp = projection * camera.toMatrix() * transform.toMatrix() * pp;
  if(fabs(pp.w())<1e-3) pp.setW(1e-2);

  QVector4D tp(v.p, 1.0);
  tp = camera.toMatrix() * transform.toMatrix() * tp;
  v.tp.setX(tp.x()); v.tp.setY(tp.y()); v.tp.setZ(tp.z());

  QVector4D wp(v.p, 1.0);
  wp = transform.toMatrix() * wp;

  VertexInfo nv=v;
  nv.n = transform.rotation().toRotationMatrix().transposed()*nv.n;
//  nv.n.setZ(-nv.n.z());
  nv.pp.setX(pp.x()/pp.w());
  nv.pp.setY(pp.y()/pp.w());
  nv.pp.setZ(pp.z()/pp.w());
  nv.pp.setW(pp.w());

  nv.wp=QVector3D(wp);
  return nv;
}

void CPURenderer::FragmentShader(DepthFragment &frag)
{
  const float ar=0.2;
  const float dr=0.3;
  const float sr=0.5;
  GI geo = frag.geo;
  frag.color = geo->ambient;
  frag.v.n.normalize();
  for(int i=0; i<lights.size(); i++)
  {
    QVector3D lightDir = (lights[i].pos-frag.v.wp).normalized();
    QVector3D viewDir = (camera.translation()-frag.v.wp).normalized();
    QVector3D reflectDir = lightDir+2*(frag.v.n-lightDir);
    QVector3D h = ((viewDir+lightDir)/2).normalized();
    ColorPixel a = geo->ambient;
    float dd = QVector3D::dotProduct(lightDir, frag.v.n.normalized())*dr;
    if(dd<0) dd=-dd;
    ColorPixel d = geo->diffuse;
    if(geo->text>=0)
    {
      d=Sample(geo->text, frag.v.tc.x(), frag.v.tc.y(), true);
      a=d;
    }
    a=a*ar;
    d=d*dd;
    float ss = QVector3D::dotProduct(reflectDir, viewDir);
    if(ss<0) ss=0; ss=pow(ss, 16);
    if(QVector3D::dotProduct(h, frag.v.n.normalized())<0) ss/=2.0;
    if(ss<0) ss=0;
    ColorPixel s = geo->specular*ss;
    frag.color.r = (a.r+d.r+s.r)*lights[i].color.x();
    frag.color.g = (a.g+d.g+s.g)*lights[i].color.y();
    frag.color.b = (a.b+d.b+s.b)*lights[i].color.z();
    frag.color.a = (a.a*ar+d.a*dr+s.a*sr);
//    frag.color=d;
//    frag.lightDir=lightDir;
  }
//  frag.color.a=1.0;
//  frag.color=(frag.v.tp+QVector3D(1.0, 1.0, 1.0))/2;
//  frag.color=QVector3D(frag.v.tc, 1.0);
}
