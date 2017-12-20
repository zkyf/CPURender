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
    s << "  vecs #" << i << ":" << g.vecs[i].p  << g.vecs[i].pp << g.vecs[i].n << g.vecs[i].tp << endl;
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

  projection.lookAt(camera.translation(), camera.translation()+camera.forward(), camera.up());
  //qDebug() << "projection" << projection;
  projection.perspective(60, 1.0, 0.1, 100.0);
//  projection.ortho(-1, 1, -1, 1, 0, 100);
  //qDebug() << "projection" << projection;
  nearPlane=0.1;
  farPlane = 100.0;
}

CPURenderer::~CPURenderer()
{
  delete[] stencilBuffer;
}

uchar* CPURenderer::Render()
{
  float epsy = 1.0/size.height();
  float epsx = 1.0/size.width();
  colorBuffer.Clear();
  depthBuffer.Clear();

  //qDebug() << "CPURenderer Start";
  // vertex shader
  {
    geos = input;
    //qDebug() << "vertex shader";
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      qDebug() << "before:" << *i;
      for(int vi=0; vi<i->vecs.size(); vi++)
      {
        i->vecs[vi]=VertexShader(i->vecs[vi]);
      }

      qDebug() << "after:" << *i;
    }
    //qDebug() << "vertex shader ended";
  }

  // light transform
  {
    for(int i=0; i<lights.size(); i++)
    {
      if(lights[i].type==Light::Point)
      {
        VertexInfo light(lights[i].pos);
        light=VertexShader(light);
        lights[i].tp=light.tp;
        qDebug() << "light #" << i << " @ " << lights[i].tp;
      }
    }
  }

  // pre-clipping
  {
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      qDebug() << "pre-clipping z<0";
      QVector<QVector3D> vecs;
      for(int v=0; v<i->vecs.size(); v++)
      {
        vecs.push_back(i->vecs[v].tp);
      }
      Clip(QVector4D(0, 0, 1, 0), false, vecs, i);
      qDebug() << "after: " << *i;
    }
  }

  // re-vertex-shader
  {
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      qDebug() << "re-vertex-shader before:" << *i;
      for(int vi=0; vi<i->vecs.size(); vi++)
      {
        i->vecs[vi]=VertexShader(i->vecs[vi]);
      }

      qDebug() << "re-vertex-shader after:" << *i;
    }
  }

  // geometry shader
//  {
//    //qDebug() << "geometry shader";
//    int geosNum = geos.size();
//    int count = 0;
//    for(GI i= geos.begin(); count<geosNum; i++)
//    {
//      GeometryShader(*i);
//    }
//    //qDebug() << "geometry shader ended";
//  }

  {
    qDebug() << "new clipping";
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      qDebug() << "before: " << *i;

      {
        qDebug() << "x>-1";
        QVector<QVector3D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(1, 0, 0, 1), true, vecs, i);
      }

      {
        qDebug() << "x<1";
        QVector<QVector3D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(-1, 0, 0, 1), true, vecs, i);
      }

      {
        qDebug() << "y>-1";
        QVector<QVector3D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, 1, 0, 1), true, vecs, i);
      }

      {
        qDebug() << "y<1";
        QVector<QVector3D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, -1, 0, 1), true, vecs, i);
      }

      {
        qDebug() << "z>-1";
        QVector<QVector3D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, 0, 1, 1), true, vecs, i);
      }

      {
        qDebug() << "z<1";
        QVector<QVector3D> vecs;
        for(int v=0; v<i->vecs.size(); v++)
        {
          vecs.push_back(i->vecs[v].pp);
        }
        Clip(QVector4D(0, 0, -1, 1), true, vecs, i);
      }

      qDebug() << "after: " << *i;
    }
  }

  // calculate dz and dy
  for(GI i=geos.begin(); i!=geos.end(); i++)
  {
    if(i->vecs.size()<3) continue;
    QVector3D v0=i->vecs[0].pp;
    QVector3D v1=i->vecs[1].pp;
    QVector3D v2=i->vecs[2].pp;
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
    //qDebug() << "geo " << *i;
    //qDebug() << "dz=" << i->dz;
  }

  // generate edge list
  QVector<EdgeListItem> edgeList;
  int nowe;
  {
    qDebug() << "regional scanline";
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

  qDebug() << "generated edge list in total " << edgeList.size();

  // new rasterization
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
    while(nowe>=0)
    {
      // find points for initial scanline

      qDebug() << "nowe=" << nowe << " on " << edgeList[nowe].geo->name << " tid=" << edgeList[nowe].tid << " bid=" << edgeList[nowe].bid;
      GI geo=edgeList[nowe].geo;
      int tid=edgeList[nowe].tid;
      int bid=edgeList[nowe].bid;
      if(fabs(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y())<epsy)
      {
        nowe--;
        continue;
      }
      if(geo->vecs[tid].pp.y()>=topy-epsy && geo->vecs[bid].pp.y()<=topy+epsy)
      {
        float r=(topy-geo->vecs[bid].pp.y())/(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y());

        ScanlinePoint np(r*geo->vecs[tid].pp+(1-r)*geo->vecs[bid].pp);
        np.n = r*geo->vecs[tid].n+(1-r)*geo->vecs[bid].n;
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
      float y=ToProjY(yy);
      for(GI i=geos.begin(); i!=geos.end(); i++)
      {
        i->inout=false;
      }

      qDebug() << yy << " : scanline.size()=" << scanline.size();
      for(int pid=0; pid<scanline.size(); pid++)
      {
        qDebug() << "pid #" << pid << " p=" << scanline[pid] << " of geometry:" << scanline[pid].geo->name << edgeList[scanline[pid].e].tid << edgeList[scanline[pid].e].bid;
      }

      for(int pid=0; pid<scanline.size(); pid++)
      {
        if(pid>0 && pid<scanline.size()-1 && (scanline[pid-1].geo==scanline[pid].geo && scanline[pid+1].geo==scanline[pid].geo) && (fabs(scanline[pid-1].x()-scanline[pid].x())<epsx || fabs(scanline[pid+1].x()-scanline[pid].x())<epsx))
        {
          qDebug() << "pid=" << pid << " skipped";
          continue;
        }

        ScanlinePoint& p=scanline[pid];
        p.geo->inout=!p.geo->inout;

        if(p.geo->inout)
        {
          if(pid==scanline.size()-1 || p.geo!=scanline[pid+1].geo)
          {
            qDebug() << "Error: Odd number of scan points @ yy=" << yy << " p=" << p << " of geometry:";
            qDebug() << *p.geo;
          }

          int nextxx;
          if(pid<scanline.size()-1) nextxx=ToScreenX(scanline[pid+1].x());
          else nextxx=size.width();
          if(nextxx>=size.width()) nextxx=size.width();
          float nowz=p.z();
          for(int xx=ToScreenX(p.x()); xx<nextxx; xx++)
          {
            //          //qDebug() << "xx=" << xx;
            DepthFragment ng;
            ng.pos.setX(ToProjX(xx));
            ng.pos.setY(y);
            float r=(ng.pos.x()-scanline[pid].x())/(scanline[pid+1].x()-scanline[pid].x());
            //          if(yy=size.height()/2)
            //          {
            //            //qDebug() << "now geo=" << s[pid].geo->name << "xx=" << xx << "nowz=" << nowz << s[pid+1].z() << s[pid].z() << s[pid+1].x() << s[pid].x();
            //          }
            ng.pos.setZ(nowz);
            ng.geo=scanline[pid].geo;
            ng.tp=r*scanline[pid+1].tp+(1-r)*scanline[pid].tp;
            ng.normal=r*scanline[pid+1].n+(1-r)*scanline[pid].n;
            nowz+=ng.geo->dz;
            // calculate deoth value of a fragment here
            depthBuffer.buffer[xx+yy*size.width()].chain.push_back(ng);
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
            scanline.remove(pid);
            pid--;
            continue;
          }
          else
          {
            p.setX(p.x()-p.dx);
            p.setY(p.y()-2.0/size.height());
            p.setZ(p.z()+p.dx*geo->dz/(2.0/size.width())-geo->dzy);
          }
        }

        while(nowe>=0)
        {
          GI geo=edgeList[nowe].geo;
          int tid=edgeList[nowe].tid;
          int bid=edgeList[nowe].bid;
          if(fabs(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y())<epsy)
          {
            nowe--;
            continue;
          }
          if(geo->vecs[tid].pp.y()>=nexty-epsy && geo->vecs[bid].pp.y()<=nexty+epsy)
          {
            float r=(nexty-geo->vecs[bid].pp.y())/(geo->vecs[tid].pp.y()-geo->vecs[bid].pp.y());

            ScanlinePoint np(r*geo->vecs[tid].pp+(1-r)*geo->vecs[bid].pp);
            np.n = r*geo->vecs[tid].n+(1-r)*geo->vecs[bid].n;
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
      }
    }
  }

//  // rasterization
//  {
//    //qDebug() << "rasterization";

//    for(GI i=geos.begin(); i!=geos.end(); i++)
//    {
//      i->Top(); i->Bottom();
//    }
//    qSort(geos);

//    //qDebug() << "geos sorted";

//    float top=-1e20, bottom=1e20;
//    for(GI i=geos.begin(); i!=geos.end(); i++)
//    {
//      //qDebug() << *i;
//      if(i->vecs.size()<3)
//      {
//        continue;
//      }

//      if(i->top>top) top=i->top;
//      if(i->bottom<bottom) bottom=i->bottom;
//    }
//    int tt=ToScreenY(top);
//    if(tt<0) tt=0; if(tt>=size.height()) tt=size.height()-1;
//    int bb=ToScreenY(bottom);
//    if(bb<0) bb=0; if(bb>=size.height()) bb=size.height()-1;
//    //qDebug() << "tt=" << tt << " bb=" << bb;

//    for(int yy=tt; yy<bb && yy<size.height(); yy++)
//    {
//      float y=ToProjY(yy);

//      // generate a scanline
//      Scanline s;
//      {
//        //qDebug() << "generate scanline";
//        for(GI i=geos.begin(); i!=geos.end(); i++)
//        {
//          Scanline s4i;
//    //        //qDebug() << "  for geo@" << *i;
//          //qDebug() << "geo.top=" << ToScreenY(i->top) << " ," << "geo.bottom=" << ToScreenY(i->bottom);
//          if(ToScreenY(i->top)>yy || ToScreenY(i->bottom)<yy)
//          {
//            continue;
//          }

//          int c=0;
//          //qDebug() << "finding hp";
//          for(int j=0; j<i->vecs.size(); j++)
//          {
//            int next=(j+1)%i->vecs.size();
//            int prev=(j-1+i->vecs.size())%i->vecs.size();
//            //qDebug() << "cond=" << (y-i->vecs[j].pp.y())*(i->vecs[next].pp.y()-y);
//            if((y-i->vecs[j].pp.y())*(i->vecs[next].pp.y()-y)>=0)
//            {
//              if(fabs(i->vecs[j].pp.y()-i->vecs[next].pp.y())<1e-3)
//              {
//                continue;
//              }

//              c++;
//              float r=(i->vecs[j].pp.y()-y)/(i->vecs[j].pp.y()-i->vecs[next].pp.y());
//              ScanlinePoint np(r*i->vecs[next].pp+(1-r)*i->vecs[j].pp);

//              np.tp=r*i->vecs[next].tp+(1-r)*i->vecs[j].tp;
//              np.n=r*i->vecs[next].n+(1-r)*i->vecs[j].n;
//              np.geo=i;
//              np.hp=c;

//              if(i->vecs[j].pp.x()<=i->vecs[next].pp.x())
//              {
//                np.prev=j;
//                np.next=next;
//              }
//              else
//              {
//                np.prev=next;
//                np.next=j;
//              }
//              s4i.push_back(np);
//            }
//          }

//          qSort(s4i);
//          int cc=1;
//          for(int j=0; j<s4i.size(); j++)
//          {
//            s4i[j].hp=cc;
//            if(j<s4i.size()-1 && j>0 && (fabs(s4i[j].x()-s4i[j+1].x())<1e-3 || fabs(s4i[j].x()-s4i[j-1].x())<1e-3))
//            {
//              if(cc%2==0)
//              {
//                if(j<s4i.size()-1)
//                {
//                  continue;
//                }
//              }
//            }
//            s.push_back(s4i[j]);
//            cc++;
//          }
//        }
//      }

//      qSort(s);

//      //qDebug() << "sl @yy=" << yy << " y=" << y << ":" << s;

//      // fill depth values
//      //qDebug() << "fill depth values";
//      QSet<GI> nowPoly;
//      for(int pid=0; pid<s.size(); pid++)
//      {
//        // fill depth buffer here;

//        //qDebug() << "hp #" << pid;

//        if(s[pid].hp%2==1)
//        {
//          nowPoly.insert(s[pid].geo);
//        }
//        else
//        {
//          nowPoly.remove(s[pid].geo);
//          continue;
//        }

//        float nn;
//        if(pid<s.size()-1)
//        {
//          nn=s[pid+1].x();
//        }
//        else
//        {
//          nn=1.0;
//        }
//        float nowz=s[pid].z();

//        int xx = ToScreenX(s[pid].x());
//        if(xx<0)
//        {
//          nowz+=s[pid].geo->dz/size.width()*2*(-xx);
//          xx=0;
//        }
////            //qDebug() << "depthfragment pushed @" << xx << ", " << yy;
//        for(; ToProjX(xx)<nn && xx<size.width(); xx++)
//        {
////          //qDebug() << "xx=" << xx;
//          DepthFragment ng;
//          ng.pos.setX(ToProjX(xx));
//          ng.pos.setY(y);
//          float r=(ng.pos.x()-s[pid].x())/(s[pid+1].x()-s[pid].x());
////          if(yy=size.height()/2)
////          {
////            //qDebug() << "now geo=" << s[pid].geo->name << "xx=" << xx << "nowz=" << nowz << s[pid+1].z() << s[pid].z() << s[pid+1].x() << s[pid].x();
////          }
//          ng.pos.setZ(nowz);
//          ng.geo=s[pid].geo;
//          ng.tp=r*s[pid+1].tp+(1-r)*s[pid].tp;
//          ng.normal=r*s[pid+1].n+(1-r)*s[pid].n;
//          nowz+=ng.geo->dz/size.width()*2;
//          // calculate deoth value of a fragment here
//          depthBuffer.buffer[xx+yy*size.width()].chain.push_back(ng);
//        }
//      }
//    }

//    //qDebug() << "rasterization ended";
//  }

  // fragment shader
  {
    //qDebug() << "fragment shader";
    for(int i=0; i<size.width()*size.height(); i++)
    {
      for(int j=0; j<depthBuffer.buffer[i].chain.size(); j++)
      {
        FragmentShader(depthBuffer.buffer[i].chain[j]);
//        //qDebug() << "color @" << i << "=" << depthBuffer.buffer[i].chain[j].color;
      }
    }
    //qDebug() << "fragment shader ended";
  }

  // per sample operations
  {
    qDebug() << "per sample operation";
    for(int i=0; i<size.width()*size.height(); i++)
    {
      qSort(depthBuffer.buffer[i].chain);
      colorBuffer.buffer[i] = ColorPixel(1.0, 1.0, 1.0, 0.0);
      for(int j=0; j<depthBuffer.buffer[i].chain.size(); j++)
      {
        colorBuffer.buffer[i] = colorBuffer.buffer[i]+depthBuffer.buffer[i].chain[j].color;
        if(i/size.width()==size.height()/2)
        {
//          qDebug() << "depthBuffer @ (" << i%size.width() << ", " << i/size.height() << ")[" << j << "]=" << depthBuffer.buffer[i].chain[j].color;
        }
        if(colorBuffer.buffer[i].a >=0.99) // not transparent
        {
          break;
        }
      }
//      colorBuffer.buffer[i] = colorBuffer.buffer[i]+ColorPixel(1.0, 1.0, 1.0, 0.0);
//      qDebug() << "colorBuffer @ (" << i%size.width() << ", " << i/size.height() << ")=" << colorBuffer.buffer[i] << depthBuffer.buffer[i].chain.size();
    }

    //qDebug() << "center pixel:";
    int xc = size.width()/2+10;
    int yc = size.height()/2;
    for(int j=0; j<depthBuffer.buffer[xc+yc*size.width()].chain.size(); j++)
    {
      //qDebug() << "frag #" << j << ":" << depthBuffer.buffer[xc+yc*size.width()].chain[j].pos << depthBuffer.buffer[xc+yc*size.width()].chain[j].color;
    }

    //qDebug() << "per sample operation ended";
  }

  uchar* result = colorBuffer.ToUcharArray();

  return result;
}

VertexInfo CPURenderer::VertexShader(VertexInfo v)
{
  QVector4D pp(v.p, 1.0);
  pp = projection * camera.toMatrix() * transform.toMatrix() * pp;
  if(fabs(pp.w())<1e-3) pp.setW(1e-2);

  QVector4D tp(v.p, 1.0);
  tp = camera.toMatrix() * transform.toMatrix() * tp;
  v.tp.setX(tp.x()); v.tp.setY(tp.y()); v.tp.setZ(tp.z());

  VertexInfo nv=v;
  nv.n = transform.rotation().toRotationMatrix()*nv.n;
  nv.pp.setX(pp.x()/pp.w());
  nv.pp.setY(pp.y()/pp.w());
  nv.pp.setZ(pp.z()/pp.w());
  return nv;
}

void CPURenderer::GeometryShader(Geometry &geo)
{

}

void CPURenderer::FragmentShader(DepthFragment &frag)
{
  GI geo = frag.geo;
//  frag.color = geo->ambient;
  for(int i=0; i<lights.size(); i++)
  {
    QVector3D lightDir = (lights[i].tp-frag.tp).normalized();
    QVector3D viewDir = (-frag.tp).normalized();
    QVector3D h = ((viewDir+lightDir)/2).normalized();
    ColorPixel a = geo->ambient*0.4;
    float ss = pow(QVector3D::dotProduct(h, frag.normal), 16)*0.6;
    if(ss<0) ss=0.0;
    ColorPixel s = geo->specular*ss;
    frag.color.r = a.r+s.r;
    frag.color.g = a.g+s.g;
    frag.color.b = a.b+s.b;
    frag.color.a = a.a;
  }
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

void CPURenderer::Clip(QVector4D A, bool dir, QVector<QVector3D> g, GI i)
{
  if(g.size()<3) return;
  if(g.size()!=i->vecs.size()) return;

  bool lastStatus = (QVector4D::dotProduct(A, QVector4D(g.last(), 1.0))<0) ^ dir;
  QVector<float> result;
  for(int i=0; i<g.size(); i++)
  {
    if((QVector4D::dotProduct(A, QVector4D(g[i], 1.0))<0) ^ dir)
    {
      qDebug() << "p #" << i << g[i] << "on " << dir << "side of " << A;
      if(!lastStatus)
      {
        QVector4D t(-g[i]+g[(i+g.size()-1)%g.size()], 0.0);
        QVector4D p0(g[i], 1.0);
        float k=QVector4D::dotProduct(-A, p0)/QVector4D::dotProduct(A, t);
        qDebug() << "out-in point:" << p0+k*t << QVector4D::dotProduct(p0+k*t, A);
        result.push_back(k);
      }
      else
      {
        result.push_back(10.0);
      }
    }
    else
    {
      qDebug() << "p #" << i << g[i] << "on " << !dir << "side of " << A;
      if(lastStatus)
      {
        QVector4D t(-g[i]+g[(i+g.size()-1)%g.size()], 0.0);
        QVector4D p0(g[i], 1.0);
        float k=QVector4D::dotProduct(-A, p0)/QVector4D::dotProduct(A, t);
        qDebug() << "in-out point:" << p0+k*t << QVector4D::dotProduct(p0+k*t, A);
        result.push_back(-k);
      }
      else
      {
        result.push_back(-10);
      }
    }

    lastStatus=(QVector4D::dotProduct(A, QVector4D(g[i], 1.0))<0) ^ dir;
  }

  {
    QVector<VertexInfo> ng;
    bool dirty=false;
    for(int v=0; v<i->vecs.size(); v++)
    {
      if(result[v]>5.0)
      {
        ng.push_back(i->vecs[v]);
      }
      else if(result[v]>0)
      {
        VertexInfo nf;
        nf. p=i->vecs[v]. p+result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()]. p-i->vecs[v]. p);
        nf.pp=i->vecs[v].pp+result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()].pp-i->vecs[v].pp);
        nf.tp=i->vecs[v].tp+result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()].tp-i->vecs[v].tp);
        nf. n=i->vecs[v]. n+result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()]. n-i->vecs[v]. n);
        ng.push_back(nf);

        ng.push_back(i->vecs[v]);
        dirty=true;
      }
      else if(result[v]>-1)
      {
        VertexInfo nf;
        nf. p=i->vecs[v]. p-result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()]. p-i->vecs[v]. p);
        nf.pp=i->vecs[v].pp-result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()].pp-i->vecs[v].pp);
        nf.tp=i->vecs[v].tp-result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()].tp-i->vecs[v].tp);
        nf. n=i->vecs[v]. n-result[v]*(i->vecs[(v+i->vecs.size()-1)%i->vecs.size()]. n-i->vecs[v]. n);
        ng.push_back(nf);
        dirty=true;
      }
    }
    if(dirty) i->vecs=ng;
  }
}
