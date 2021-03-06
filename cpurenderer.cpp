#include "cpurenderer.h"

extern "C" void CudaIntersect(CudaRay ray);
extern "C" void CudaInit(CudaGeometry* geos, int _n, int* lightList, int _ln, double* h, double* randNum, int _randN);
extern "C" void CudaEnd();
extern "C" void CudaRender(int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right, CudaVec* buffer);
extern "C" void CudaGetRayTest(int xx, int yy, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right);
extern "C" void CudaVectorTest();
extern "C" void CudaKdTreeTest(CudaGeometry* geolist, int n);

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

bool operator< (const EdgeListItem& a, const EdgeListItem& b)
{
  if(a.geo->vecs[a.tid].pp.y()!=b.geo->vecs[b.tid].pp.y())
    return a.geo->vecs[a.tid].pp.y()<b.geo->vecs[b.tid].pp.y();
  else
    return a.geo->vecs[a.bid].pp.y()<b.geo->vecs[b.bid].pp.y();
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
  stencilBuffer(nullptr),
  hits(nullptr)
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

  srand(time(0));

//  CudaVectorTest();
}

CPURenderer::~CPURenderer()
{
  delete[] stencilBuffer;
  delete[] hits;

  CudaEnd();
}

uchar* CPURenderer::Render()
{
  double epsy = 0.1/size.height();
  double epsx = 0.1/size.width();
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
      Clip(QVector4D(0, 0, 1, 0), false, vecs, i, dirty, false);
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
    Mat A(2, 2, CV_64F, Scalar::all(0));
    A.at<double>(0, 0)=v2.x()-v0.x();
    A.at<double>(0, 1)=v2.y()-v0.y();
    A.at<double>(1, 0)=v2.x()-v1.x();
    A.at<double>(1, 1)=v2.y()-v1.y();
    Mat b(2, 1, CV_64F, Scalar::all(0));
    b.at<double>(0, 0)=v2.z()-v0.z();
    b.at<double>(1, 0)=v2.z()-v1.z();
    Mat dxy=A.inv()*b;

    i->dz = dxy.at<double>(0, 0)/size.width()*2;
    i->dzy = dxy.at<double>(1, 0)/size.height()*2;
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
//      if(geo->vecs[edgeList[nowe].tid].pp.y()>1.0 && geo->vecs[edgeList[nowe].bid].pp.y()>1.0)
      if(ToScreenY(geo->vecs[edgeList[nowe].tid].pp.y())<0 && ToScreenY(geo->vecs[edgeList[nowe].bid].pp.y())<0)
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
    double topy=ToProjY(topyy);

    Scanline scanline;
    //qDebug() << "topy=";
    while(nowe>=0)
    {
      // find points for initial scanline

//      qDebug() << "nowe=" << nowe << " on " << edgeList[nowe].geo->name << " tid=" << edgeList[nowe].tid << " bid=" << edgeList[nowe].bid;
      GI geo=edgeList[nowe].geo;
      int tid=edgeList[nowe].tid;
      int bid=edgeList[nowe].bid;
      if(ToScreenY(geo->vecs[tid].pp.y())==ToScreenY(geo->vecs[bid].pp.y()))
      {
//        qDebug() << "ignore horizontal edge";
        nowe--;
        continue;
      }
      //qDebug() << "geo->vecs[tid].pp.y()-topy=" << geo->vecs[tid].pp.y()-topy << "geo->vecs[bid].pp.y()-nexty=" << geo->vecs[bid].pp.y()-topy;
      if(ToScreenY(geo->vecs[tid].pp.y())<=topyy && ToScreenY(geo->vecs[bid].pp.y())>=topyy)
      {
        double r=(topyy-ToScreenY(geo->vecs[bid].pp.y()))*1.0/(ToScreenY(geo->vecs[tid].pp.y())-ToScreenY(geo->vecs[bid].pp.y()));

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
      double y=ToProjY(yy);
      for(GI i=geos.begin(); i!=geos.end(); i++)
      {
        i->inout=false;
      }

//      qDebug() << yy << " : scanline.size()=" << scanline.size();
      for(int pid=0; pid<scanline.size(); pid++)
      {
//        qDebug() << "pid #" << pid << " p=" << scanline[pid] << " of geometry:" << scanline[pid].geo->name << edgeList[scanline[pid].e].tid << edgeList[scanline[pid].e].bid << scanline[pid].dx*size.height()/2;
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
            qDebug() << "Error: Odd number of scan points @ yy=" << yy << " p=" << p << " of geometry:";
            error=true;
            qDebug() << *p.geo;
          }

          int nextxx;
          if(pid<scanline.size()-1) nextxx=ToScreenX(scanline[pid+1].x());
          else nextxx=size.width();
          if(error) nextxx=ToScreenX(scanline[pid].x())+1;
          if(nextxx>=size.width()) nextxx=size.width();
          double nowz=p.z();
          for(int xx=ToScreenX(p.x()); xx<nextxx; xx++)
          {
            //          ////qDebug() << "xx=" << xx;
            DepthFragment ng;
            ng.pos.setX(ToProjX(xx));
            ng.pos.setY(y);
            ng.index=xx+yy*size.width();
            double r=(ng.pos.x()-scanline[pid].x())/(scanline[pid+1].x()-scanline[pid].x());
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
//          if(pid<scanline.size()-1 && (pid==scanline.size()-2 || (scanline[pid+1].geo==scanline[pid].geo && scanline[pid+2].geo!=scanline[pid+1].geo)))
          {
            scanline[pid].geo->inout=!scanline[pid].geo->inout;
          }
        }
      }

      // generate new scanline
      {
        double nexty=ToProjY(yy+1);
        for(int pid=0; pid<scanline.size(); pid++)
        {
          ScanlinePoint& p=scanline[pid];
          GI geo=p.geo;
          EdgeListItem& e=edgeList[p.e];
          if(ToScreenY(geo->vecs[e.bid].pp.y())<=yy+1)
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

            double r = (p.y()-geo->vecs[e.bid].pp.y()) / (geo->vecs[e.tid].pp.y()-geo->vecs[e.bid].pp.y());
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
          if(ToScreenY(geo->vecs[tid].pp.y())<=yy+1 && ToScreenY(geo->vecs[bid].pp.y())>=yy+1)
          {
//            double r=(nexty-geo->vecs[bid].pp.y())/(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y());
            double r=(yy+1-ToScreenY(geo->vecs[bid].pp.y()))*1.0/(ToScreenY(geo->vecs[tid].pp.y())-ToScreenY(geo->vecs[bid].pp.y()));

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
      }
    }
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
  input.last().id=input.size()-1;
  if(geo.emission.Strength()>0.1)
  {
    lightGeoList.push_back(input.size()-1);
  }
}

int CPURenderer::ToScreenX(double x)
{
  return (int)((x+1.0)/2*(size.width()-1));
}

int CPURenderer::ToScreenY(double y)
{
  return (int)((-y+1.0)/2*(size.height()-1));
}

double CPURenderer::ToProjX(int x)
{
  return x*1.0/(size.width()-1)*2-1.0;
}

double CPURenderer::ToProjY(int y)
{
  return -(y*1.0/(size.height()-1)*2-1.0);
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

void CPURenderer::Clip(QVector4D A, bool dir, QVector<QVector4D> g, GI i, bool &dirty, bool pers)
{
  if(g.size()<3) return;
  if(g.size()!=i->vecs.size()) return;

  for(int vid=0; vid<g.size(); vid++)
  {
    g[vid].setW(1.0);
  }

  bool lastStatus = (QVector4D::dotProduct(A, g.last())<0) ^ dir;
  QVector<double> result;
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
        double k=QVector4D::dotProduct(-A, p0)/QVector4D::dotProduct(A, t);
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
        double k=QVector4D::dotProduct(-A, p0)/QVector4D::dotProduct(A, t);
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
        nf = VertexInfo::Intersect(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()], i->vecs[v], result[v], pers);
        ng.push_back(nf);

        ng.push_back(i->vecs[v]);
        dirty=true;
      }
      else if(result[v]>-1)
      {
        VertexInfo nf;
        nf = VertexInfo::Intersect(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()], i->vecs[v], -result[v], pers);
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

ColorPixel CPURenderer::Sample(int tid, double u, double v, bool bi)
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

    double uu=u*(w-1);
    double vv=(1-v)*(h-1);

    double ur=uu-ll;
    double vr=vv-tt;

    double ltr = (1-ur)*(1-vr);
    double rtr = ur*(1-vr);
    double lbr = (1-ur)*vr;
    double rbr = ur*vr;

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
  if(fabs(pp.w())<1e-5) pp.setW(1e-5);

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
  nv.pp.setW(1.0/pp.w());

  nv.ptc = nv.tc/pp.w();
  wp /=pp.w();
  nv.wp=QVector3D(wp);
  return nv;
}

void CPURenderer::FragmentShader(DepthFragment &frag)
{
  const double ar=0.2;
  const double dr=0.3;
  const double sr=0.5;
  GI geo = frag.geo;
  frag.color = geo->ambient;
  frag.v.n.normalize();
  for(int i=0; i<lights.size(); i++)
  {
    QVector3D lightDir = (lights[i].pos-frag.v.wp/frag.v.pp.w()).normalized();
    QVector3D viewDir = (camera.translation()-frag.v.wp/frag.v.pp.w()).normalized();
    float rl = QVector3D::dotProduct(lightDir, frag.v.n);
    QVector3D reflectDir = lightDir+2*(rl*frag.v.n-lightDir);
    QVector3D h = (viewDir+lightDir).normalized();
    ColorPixel a = geo->ambient;
    double dd = QVector3D::dotProduct(h, frag.v.n.normalized())*dr;
    if(dd<0) dd=-dd;
    ColorPixel d = geo->diffuse;
    if(geo->text>=0)
    {
      d=Sample(geo->text, frag.v.tc.x(), frag.v.tc.y(), true);
      a=d;
    }
    a=a*ar;
    d=d*dd;
    double ss = QVector3D::dotProduct(reflectDir, viewDir);
    if(ss<0) ss=0; ss=pow(ss, 16);
    ColorPixel s = geo->specular*ss;
    if(geo->stext>=0)
    {
      ColorPixel sm = Sample(geo->stext, frag.v.tc.x(), frag.v.tc.y(), true);
      s=s*sm;
    }
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


uchar* CPURenderer::MonteCarloRender()
{
  colorBuffer.Clear();
  depthBuffer.Clear();
  if(hits)
  {
    delete[] hits;
    hits = nullptr;
  }

  CudaGeometry* geos=nullptr;
  if(input.size()>0)
  {
    geos = new CudaGeometry[input.size()];
    for(int i=0; i<input.size(); i++)
    {
      CudaGeometry newg;
      newg.index=i;
      newg.diffuse = CudaVec(input[i].diffuse.r, input[i].diffuse.g, input[i].diffuse.b);
      newg.emission = CudaVec(input[i].emission.r, input[i].emission.g, input[i].emission.b);
      newg.specular = CudaVec(input[i].specular.r, input[i].specular.g, input[i].specular.b);
      newg.reflectr = input[i].reflectr;
      newg.refractr = input[i].refractr;
      newg.ni = input[i].ni;
      if(input[i].reflectr>1e-3)
      {
//        qDebug() << "ReflectGeo #" << i << input[i].reflectr;
      }

      for(int j=0; j<input[i].vecs.size(); j++)
      {
        newg.vecs[j].p = CudaVec4(input[i].vecs[j].p.x(), input[i].vecs[j].p.y(), input[i].vecs[j].p.z(), 1.0);
        newg.vecs[j].n = CudaVec4(input[i].vecs[j].n.x(), input[i].vecs[j].n.y(), input[i].vecs[j].n.z(), 1.0);
        newg.vecs[j].geo = i;
      }
      geos[i] = newg;
    }
    hits = new double[input.size()];
    double* randNum = new double[10000];
    for(int i=0; i<10000; i++)
    {
      randNum[i] = (double)(rand()%nos)/nos;
    }
    printf("HOST: sizeof(CudaGeometry)=%d, sizeof(CudaVec)=%d n=%d\n", sizeof(CudaGeometry), sizeof(CudaVec), input.size());
//    qDebug() << "lightGeoList = " << lightGeoList;

    CudaInit(geos, input.size(), lightGeoList.data(), lightGeoList.size(), hits, randNum, 10000);
  }

  CudaVec* buffer = new CudaVec[size.width()*size.height()+10];
  CudaRender(size.width(), size.height(), CudaVec(camera.translation()), CudaVec(camera.up()), CudaVec(camera.forward()), CudaVec(camera.right()), buffer);
  for(int yy=0; yy<size.height(); yy++)
  {
    for(int xx=0; xx<size.width(); xx++)
    {
      int index = yy*size.width()+xx;
      colorBuffer.buffer[index].r = buffer[index].x;
      colorBuffer.buffer[index].g = buffer[index].y;
      colorBuffer.buffer[index].b = buffer[index].z;
      colorBuffer.buffer[index].a = 1.0;
    }
  }
  if(geos)
  {
    delete[] geos;
  }
  delete[] buffer;
  return colorBuffer.ToUcharArray();

//  for(GI i=input.begin(); i!=input.end(); i++)
//  {
//    i->gi = i;
//  }
  kdtree.SetTree(input);
//  kdtree.Print();
  const int xspp = 1; // subpixels on x
  const int yspp = 1; // subpixels on y

  qDebug() << "input.size() = " << input.size();
  for(int yy=0; yy<size.height(); yy++)
  {
    bool printed=false;
    for(int xx=0; xx<size.width(); xx++)
    {
      colorBuffer.buffer[yy*size.width()+xx] = ColorPixel(0.0, 0.0, 0.0, 0.0);
//      if(yy!=79 || xx!=71) continue;
//      for(int sp=0; sp<=spsp; sp++)
      {
        Ray ray = GetRay(xx, yy);
//        colorBuffer.buffer[yy*size.width()+xx] = (ray.n+QVector4D(1.0, 1.0, 1.0, 1.0))/2;
//        colorBuffer.buffer[yy*size.width()+xx].a = 1.0;
//        continue;
//          for(GI geo=input.begin(); geo!=input.end(); geo++)
        {

          KDTree::IR ir = kdtree.Intersect(ray);
//            qDebug() << ir.valid;
          if(!ir.valid) continue;
          if(!printed)
          {
            qDebug() << QVector2D(xx, yy) << " : " <<  ray;
            printed=true;
          }

          VertexInfo vx = ray.IntersectGeo(ir.geo);
          if(vx.valid)
          {
////              qDebug() << vx.p;
//            DepthFragment frag;
//            frag.geo = vx.geo;
////              qDebug() << "geo=" << frag.geo->name;
////              frag.color = vx.n/2 + QVector3D(0.5, 0.5, 0.5);
//            frag.color = MonteCarloSample(vx, ray, 0);
//            frag.pos.setZ((vx.p-camera.translation()).length());
//            depthBuffer.buffer[yy*size.width()+xx].chain.push_back(frag);
////              qDebug() << "done";
            colorBuffer.buffer[yy*size.width()+xx] = MonteCarloSample(vx, ray, 0);
          }
        }
      }

//      qSort(depthBuffer.buffer[yy*size.width()+xx].chain);
    }
  }

  qDebug() << "render ended";
  return colorBuffer.ToUcharArray();
}

Ray CPURenderer::GetRay(int xx, int yy)
{
  float vh = M_PI/3;
  float vw = vh/size.height() * size.width();
  float vhp = vh/size.height();
  float vwp = vw/size.width();

  Ray ray;
  ray.o = camera.translation();
  ray.o.setW(1);
  ray.n = (camera.forward()+camera.right()*tan(vwp*xx-vw/2)+camera.up()*tan(vhp*(size.height() - yy - 1)-vh/2)).normalized();

  return ray;
}

ColorPixel CPURenderer::MonteCarloSample(const VertexInfo o, const Ray &i, int length, bool debuginfo)
{
  const double drr = 10.0;
  if(debuginfo)
  {
    printf("test\n");
    fflush(stdout);
    qDebug() << "";
    qDebug() << "MonteCarloSample : #" << length << i;
    qDebug() << o.p;

    if(length<=nop)
    {
      CudaRay cray;
      cray.FromRay(i);
      CudaIntersect(cray);
    }
  }
  if(!o.valid) return ColorPixel();

//  return o.geo->diffuse;

  ColorPixel result = (0.0, 0.0, 0.0, 0.0);

  // emmision
  ColorPixel emitLight = o.geo->emission;
  if(emitLight.Strength()>0.1) return emitLight;

  if(length>nop)
  {
    if(debuginfo)
    {
      qDebug() << "Ray too long" << emitLight;
    }

    // can use precomputed light visible info instead
    int hitcount=0;
    ColorPixel lightHit;
    for(int ii=0; ii<lightGeoList.size(); ii++)
    {
      Geometry& geo=input[lightGeoList[ii]];
      for(int j=0; j<nol; j++)
      {
        VertexInfo v=geo.Sample();
        if(debuginfo)
        {
          qDebug() << "test light hit #" << i << geo.name << " @ " << v.p << v.valid;
        }
        Ray out;
        out.o = o.p; out.o.setW(1.0);
        out.ni=i.ni;
        QVector3D nn((v.p-o.p).normalized());
        out.n=QVector4D(nn, 0.0);


//        CudaRay cray; cray.FromRay(out);
//        CudaIntersect(cray);
//        double mind = 1e20;
//        int ming = -1;
//        for(int iii=0; iii<input.size(); iii++)
//        {
//          if(hits[iii]>1e-3 && hits[iii]<mind)
//          {
//            mind=hits[iii];
//            ming=iii;
//          }
//        }
//        if(debuginfo)
//        {
//          qDebug() << "mind=" << mind << "ming=" << ming << "lightGeoList[ii]=" << lightGeoList[ii];
//          qDebug() << "hits=";
//          for(int i=0; i<input.size(); i++)
//          {
//            if(hits[i]>0) qDebug() << "#" << i << hits[i];
//          }
//        }
//        if(ming>=0)
//        {
//          if(ming==lightGeoList[ii])
//          {
//            float r=mind*mind;
//            if(r<fabs(QVector3D::dotProduct(o.n, nn))) r=fabs(QVector3D::dotProduct(o.n, nn));
//            if(debuginfo)
//            {
//              qDebug() << "before blend lightHit=" << lightHit << "r=" << r;
//            }
//            lightHit.LinearBlend(geo.emission*fabs(QVector3D::dotProduct(o.n, nn))/r);
//            if(debuginfo)
//            {
//              qDebug() << "after lightHit=" << lightHit;
//            }
//            hitcount++;
//          }
//        }

        KDTree::IR ir = kdtree.Intersect(out);
        if(debuginfo)
        {
          qDebug() << "test light hit ir" << ir.valid << ir.geo;
          if(ir.valid)
          {
            qDebug() << ir.geo->name << ir.d << ir.geo->emission;
          }
        }
        if(ir.valid && ir.geo->id==geo.id)
        {
          if(debuginfo)
          {
            qDebug() << "Hit light" << geo.name;
          }
          float r=ir.d*ir.d;
          if(r<fabs(QVector3D::dotProduct(o.n, nn))) r=fabs(QVector3D::dotProduct(o.n, nn));
          if(debuginfo)
          {
            qDebug() << "before blend lightHit=" << lightHit << "r=" << r;
          }
          lightHit.LinearBlend(geo.emission*fabs(QVector3D::dotProduct(o.n, nn))/r);
          if(debuginfo)
          {
            qDebug() << "after lightHit=" << lightHit;
          }
          hitcount++;
        }
        else
        {
          if(debuginfo)
          {
            qDebug() << "Didnt hit light" << geo.name;
          }
        }


      }
    }

    if(hitcount>0)
    {
      lightHit = lightHit/hitcount;
    }
    if(debuginfo)
    {
      qDebug() << "lightHit=" << lightHit;
    }

    return lightHit*o.geo->diffuse;
  }

  // diffuse
  ColorPixel diffLight;
  if(1-o.geo->refractr-o.geo->reflectr>1e-2)
  {
    int hitcount=0;
    QVector<float> phis, thetas;
    for(int ii=0; ii<1; ii++)
    {
      float phi=rand()%nos*2*M_PI/nos;
      float theta=asin(rand()%nos*2.0/nos-1.0);
      phis.push_back(phi);
      thetas.push_back(theta);
    }
    bool hitlight=false;

    if(o.geo->emission.Strength()<=0.1)
    {
      for(int ii=0; ii<spsp; ii++)
      {
        float phi=phis[ii];
        float theta=thetas[ii];

        Ray out;
        out.o = o.p;
        out.o.setW(1.0);
        out.ni=i.ni;
        QVector3D nn(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
  //      if(QVector3D::dotProduct(nn, o.n)<0)
  //      {
  //        nn = -nn;
  //      }
        out.n=QVector4D(nn, 0.0);

//        CudaRay cray; cray.FromRay(out);
//        CudaIntersect(cray);
//        double mind = 1e20;
//        int ming = -1;
//        for(int iii=0; iii<input.size(); iii++)
//        {
//          if(hits[iii]>1e-3 && hits[iii]<mind)
//          {
//            mind=hits[iii];
//            ming=iii;
//          }
//        }
//        if(ming>=0)
//        {
//          if(input[ming].emission.Strength()>0.1 && o.geo->emission.Strength()<0.10)
//          {
//            // hit light
//            continue;
//          }
//          VertexInfo hp = out.IntersectGeo(input[ming].gi);
//          hp.geo = input[ming].gi;
//          ColorPixel tr = MonteCarloSample(hp, out, length+1, debuginfo);
//          if(fabs((hp.p-QVector3D(out.o)).length() - mind)>1e-3)
//          {
//            qDebug() << "Different result!!! ir=" << (hp.p-QVector3D(out.o)).length() << " mind=" << mind;
//          }
//          float r=mind*mind;
//          if(r<1) r=1;
//          diffLight.LinearBlend(tr/r);
//          hitcount++;
//        }


        KDTree::IR ir=kdtree.Intersect(out);
        if(!ir.valid)
        {
          out.n=-out.n;
          ir=kdtree.Intersect(out);
        }
        if(ir.valid)
        {
          if(debuginfo)
          {
            qDebug() << "Hit new geo";
          }
          VertexInfo vo = ir.hp;
          if(ir.geo->emission.Strength()>0.10 && o.geo->emission.Strength()<0.10)
          {
            hitlight=true;
            continue;
          }
          ColorPixel tr = MonteCarloSample(ir.hp, out, length+1, debuginfo);
//          ColorPixel tr = (nn+QVector3D(1.0, 1.0, 1.0))/2;
          float r=ir.d*ir.d;
          if(r<1) r=1;
          diffLight.LinearBlend(tr/r);
  //        qDebug() << phi << theta;
  //        diffLight.LinearBlend(ColorPixel(phi/M_PI/2, phi/M_PI/2, phi/M_PI/2, 1.0));
          if(debuginfo)
          {
            qDebug() << "diffLight=" << diffLight << tr << "1/r=" << 1.0/r;
          }
          hitcount++;
        }
        else
        {
          if(debuginfo)
          {
            qDebug() << "Hit nothing" << diffLight;
          }
        }


      }
    }

    if(debuginfo)
    {
      qDebug() << "light #" << lightGeoList.size();
    }

    if(hitcount>0)
    {
      diffLight = diffLight/hitcount;
    }
//    diffLight.r += 0.2;
//    diffLight.Clamp();

    if(emitLight.Strength()>0.1)
    {
      diffLight = emitLight;
    }

    diffLight = diffLight * o.geo->diffuse;

    if(debuginfo)
    {
      qDebug() << "emit=" << emitLight << "diff=" << diffLight << hitcount << "result=" << result;
    }
  }

  // specular
  ColorPixel specular;
  if(o.geo->reflectr>1e-2)
  {
    QVector4D nl=o.n*QVector4D::dotProduct(o.n, -1*i.n);
    QVector4D nr=2*nl+i.n;
    Ray out;
    out.o=o.p; out.o.setW(1);
    out.n=nr.normalized();
    out.ni=i.ni;
    KDTree::IR ir=kdtree.Intersect(out);
    if(ir.valid)
    {
      specular = MonteCarloSample(ir.hp, out, length+1, debuginfo) * o.geo->specular;
    }
  }

  // Refraction
//  ColorPixel refrection;
//  if(o.geo->tran)
//  {
//    float s1 = sqrt(1-pow(QVector4D::dotProduct(o.n,i.n), 2));
//    if(fabs(o.geo->ni-i.ni)<1e-3)
//    {

//    }
//  }

  result = diffLight;
  if(o.geo->reflectr>1e-2)
  {
    result.LinearBlend(specular*o.geo->reflectr);
  }
  result.a = 1.0;
//  result = result * o.geo->diffuse;
//  result.Clamp();

  return result;
}
