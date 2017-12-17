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

  projection.lookAt(camera.translation(), camera.translation()+camera.forward(), camera.up());
  //qDebug() << "projection" << projection;
  projection.perspective(60, 1.0, 0.1, 100.0);
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
  colorBuffer.Clear();
  depthBuffer.Clear();

  //qDebug() << "CPURenderer Start";
  // vertex shader
  {
    geos = input;
    //qDebug() << "vertex shader";
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      i->vecs.clear();
      for(int vi=0; vi<i->pos.size(); vi++)
      {
        i->vecs.push_back(VertexShader(i->pos[vi], i->norms[vi], i->texcoords[vi]));
      }
    }

    for(int i=0; i<geos.size(); i++)
    {
      //qDebug() << "after: geos#" << i;
      for(int j=0; j<geos[i].vecs.size(); j++)
      {
        //qDebug() << "  vecs#" << j << ":" << geos[i].vecs[j];
      }
    }

    //qDebug() << "vertex shader ended";
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

  // post process & clipping
  {
    //qDebug() << "post process & clipping";
    //qDebug() << "farPlane=" << farPlane << ", nearPlane=" << nearPlane;
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      //qDebug() << "now processing: " << *i;
      Geometry ng;
      ng.ambient=i->ambient;
      ng.diffuse=i->diffuse;
      ng.specular=i->specular;
      bool dirt = false;
      bool last = false; // true=in, false=out

      QVector3D p = i->vecs[0];
      //qDebug() << "#0 @" << p;
      if(p.x()>=-1 && p.x()<=1 && p.y()>=-1 && p.y()<=1 && p.z()<= 1 && p.z()>=-1)
      {
        last = true;
      }
      else
      {
        dirt=true;
        last = false;
      }
      //qDebug() << "# 0 :" << dirt << last;

      for(int jj=0; jj<i->vecs.size(); jj++)
      {
        int j=(jj+1)%i->vecs.size();
        QVector3D p = i->vecs[j];
        if(p.x()>=-1 && p.x()<=1 && p.y()>=-1 && p.y()<=1 && p.z()<= 1 && p.z()>=-1)
        {
          //qDebug() << "this==true";

          if(last==false)
          {
            //qDebug() << "last==false";
            QVector3D l=i->vecs[(j-1+i->vecs.size())%i->vecs.size()];
            QVector3D t=p;

            if(l.x()<-1)
            {
              l.setY(fabs((-1-t.x()))/fabs((l.x()-t.x()))*l.y()+fabs((l.x()+1))/fabs((l.x()-t.x()))*t.y());
              l.setZ(fabs((-1-t.x()))/fabs((l.x()-t.x()))*l.z()+fabs((l.x()+1))/fabs((l.x()-t.x()))*t.z());
              l.setX(-1);
            }
            if(l.x()>1)
            {
              l.setY(fabs((1-t.x()))/fabs((l.x()-t.x()))*l.y()+fabs((l.x()-1))/fabs((l.x()-t.x()))*t.y());
              l.setZ(fabs((1-t.x()))/fabs((l.x()-t.x()))*l.z()+fabs((l.x()-1))/fabs((l.x()-t.x()))*t.z());
              l.setX(1);
            }

            if(l.y()<-1)
            {
              l.setX(fabs((-1-t.y()))/fabs((l.y()-t.y()))*l.x()+fabs((l.y()+1))/fabs((l.y()-t.y()))*t.x());
              l.setZ(fabs((-1-t.y()))/fabs((l.y()-t.y()))*l.z()+fabs((l.y()+1))/fabs((l.y()-t.y()))*t.z());
              l.setY(-1);
            }
            if(l.y()>1)
            {
              l.setX(fabs((1-t.y()))/fabs((l.y()-t.y()))*l.x()+fabs((l.y()-1))/fabs((l.y()-t.y()))*t.x());
              l.setZ(fabs((1-t.y()))/fabs((l.y()-t.y()))*l.z()+fabs((l.y()-1))/fabs((l.y()-t.y()))*t.z());
              l.setY(1);
            }

            if(l.z()<-1)
            {
              l.setX(fabs((-1-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-(-1)))/fabs((l.z()-t.z()))*t.x());
              l.setY(fabs((-1-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-(-1)))/fabs((l.z()-t.z()))*t.y());
              l.setZ(-1);
            }
            if(l.z()>1)
            {
              l.setX(fabs((1-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-1))/fabs((l.z()-t.z()))*t.x());
              l.setY(fabs((1-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-1))/fabs((l.z()-t.z()))*t.y());
              l.setZ(1);
            }

            //qDebug() << "result l=" << l;
            ng.vecs.push_back(l);
          }

          ng.vecs.push_back(p);
          ng.norms.push_back(i->norms[j]);
          ng.texcoords.push_back(i->texcoords[j]);
          last=true;
        }
        else
        {
          //qDebug() << "this==false";
          dirt=true;
          if(last==true)
          {
            //qDebug() << "last==true";
            QVector3D l=i->vecs[(j-1+i->vecs.size())%i->vecs.size()];
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

            if(t.z()<-1)
            {
              t.setX(fabs((-1-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-(-1)))/fabs((l.z()-t.z()))*t.x());
              t.setY(fabs((-1-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-(-1)))/fabs((l.z()-t.z()))*t.y());
              t.setZ(-1);
            }
            if(t.z()>1)
            {
              t.setX(fabs((1-t.z()))/fabs((l.z()-t.z()))*l.x()+fabs((l.z()-1))/fabs((l.z()-t.z()))*t.x());
              t.setY(fabs((1-t.z()))/fabs((l.z()-t.z()))*l.y()+fabs((l.z()-1))/fabs((l.z()-t.z()))*t.y());
              t.setZ(1);
            }

            ng.vecs.push_back(t);
            //qDebug() << "add point for out-in point: " << t;
          }
          else
          {
            QVector3D l=i->vecs[(j-1+i->vecs.size())%i->vecs.size()];
            QVector3D t=p;
            QVector<float> rlist;

            // intersect with x=-1
            if((l.x()+1)*(t.x()+1)<0)
            {
              rlist.push_back(1-(t.x()+1)/(t.x()-l.x()));
            }
            // intersect with x=1
            if((l.x()-1)*(t.x()-1)<0)
            {
              rlist.push_back(1-(t.x()-1)/(t.x()-l.x()));
            }
            // intersect with y=-1
            if((l.y()+1)*(t.y()+1)<0)
            {
              rlist.push_back(1-(t.y()+1)/(t.y()-l.y()));
            }
            // intersect with y=1
            if((l.y()-1)*(t.y()-1)<0)
            {
              rlist.push_back(1-(t.y()-1)/(t.y()-l.y()));
            }

            if(rlist.size()>0)
            {
              qSort(rlist);
              for(int i=0; i<rlist.size(); i++)
              {
                QVector3D toAdd = rlist[i]*t+(1-rlist[i])*l;
                if(toAdd.x()>=-1-1e-3 && toAdd.x()<=1+1e-3 && toAdd.y()>=-1-1e-3 && toAdd.y()<=1+1e-3 && toAdd.z()>=-1-1e-3 && toAdd.z()<=1+1e-3)
                {
                  ng.vecs.push_back(toAdd);
                  //qDebug() << "add point for cross false points: " << toAdd;
                }
                else
                {
                  //qDebug() << "ignore point for cross false points: " << toAdd;
                  bool c1=toAdd.x()>=-1, c2=toAdd.x()<=1, c3=toAdd.y()>=-1, c4=toAdd.y()<=1, c5=toAdd.z()>=-1, c6=toAdd.z()<=1;
                  //qDebug() << "toAdd.x()>=-1:" << c1 << "toAdd.x()<=1:" << c2
                      //<< "toAdd.y()>=-1:" << c3 << "toAdd.y()<=1:" << c4
                      //<< "toAdd.z()>=-1:" << c5 << "toAdd.z()<=1:" << c6;
                }
              }
            }
          }
          last=false;
        }
      }

      if(dirt)
      {
//        *i=ng;
      }
    }

    for(int i=0; i<geos.size(); i++)
    {
      //qDebug() << "after: geos#" << i;
      for(int j=0; j<geos[i].vecs.size(); j++)
      {
        //qDebug() << "  vecs#" << j << ":" << geos[i].vecs[j];
      }
    }

    // calculate dz and dy
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      if(i->vecs.size()<3) continue;
      QVector3D v0=i->vecs[0];
      QVector3D v1=i->vecs[1];
      QVector3D v2=i->vecs[2];
      Mat A(2, 2, CV_32F, Scalar::all(0));
      A.at<float>(0, 0)=v2.x()-v0.x();
      A.at<float>(0, 1)=v2.y()-v0.y();
      A.at<float>(1, 0)=v2.x()-v1.x();
      A.at<float>(1, 1)=v2.y()-v1.y();
      Mat b(2, 1, CV_32F, Scalar::all(0));
      b.at<float>(0, 0)=v2.z()-v0.z();
      b.at<float>(1, 0)=v2.z()-v1.z();
      Mat dxy=A.inv()*b;

      i->dz = dxy.at<float>(0, 0);
      //qDebug() << "geo " << *i;
      //qDebug() << "dz=" << i->dz;
    }
    //qDebug() << "post process & clipping ended";
  }

  // rasterization
  {
    //qDebug() << "rasterization";

    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      i->Top(); i->Bottom();
    }
    qSort(geos);

    //qDebug() << "geos sorted";

    float top=-1e20, bottom=1e20;
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      //qDebug() << *i;
      if(i->vecs.size()<3)
      {
        continue;
      }

      if(i->top>top) top=i->top;
      if(i->bottom<bottom) bottom=i->bottom;
    }
    int tt=ToScreenY(top);
    if(tt<0) tt=0; if(tt>=size.height()) tt=size.height()-1;
    int bb=ToScreenY(bottom);
    if(bb<0) bb=0; if(bb>=size.height()) bb=size.height()-1;
    //qDebug() << "tt=" << tt << " bb=" << bb;

    for(int yy=tt; yy<bb && yy<size.height(); yy++)
    {
      float y=ToProjY(yy);
      // generate a scanline
      Scanline s;
      //qDebug() << "generate scanline";
      for(GI i=geos.begin(); i!=geos.end(); i++)
      {
        Scanline s4i;
//        //qDebug() << "  for geo@" << *i;
        //qDebug() << "geo.top=" << ToScreenY(i->top) << " ," << "geo.bottom=" << ToScreenY(i->bottom);
        if(ToScreenY(i->top)>yy || ToScreenY(i->bottom)<yy)
        {
          continue;
        }

        int c=0;
        //qDebug() << "finding hp";
        for(int j=0; j<i->vecs.size(); j++)
        {
          int next=(j+1)%i->vecs.size();
          int prev=(j-1+i->vecs.size())%i->vecs.size();
          //qDebug() << "cond=" << (y-i->vecs[j].y())*(i->vecs[next].y()-y);
          if((y-i->vecs[j].y())*(i->vecs[next].y()-y)>=0)
          {
            if(fabs(i->vecs[j].y()-i->vecs[next].y())<1e-5)
            {
              continue;
            }

            if(fabs(i->vecs[j].y()-y)<=1e-5)
            {
              if((i->vecs[prev].y()-y)*(i->vecs[next].y()-y)<0)
              {
                //qDebug() << "hp ignored @" << j << " @" << i->vecs[j];
                //qDebug() << "prev=" << prev << " , next=" << next;
                continue;
              }
            }

            c++;
            float r=(i->vecs[j].y()-y)/(i->vecs[j].y()-i->vecs[next].y());
            ScanlinePoint np(r*i->vecs[next]+(1-r)*i->vecs[j]);

            np.geo=i;
            np.hp=c;

            if(i->vecs[j].x()<=i->vecs[next].x())
            {
              np.prev=j;
              np.next=next;
            }
            else
            {
              np.prev=next;
              np.next=j;
            }
            s4i.push_back(np);
          }
        }

        qSort(s4i);
        int cc=1;
        for(int j=0; j<s4i.size(); j++)
        {
          s4i[j].hp=cc;
          if(j>0 && fabs(s4i[j].x()-s4i[j-1].x())<1e-3)
          {
//            continue;
          }
          s.push_back(s4i[j]);
          cc++;
        }
      }

      qSort(s);

      //qDebug() << "sl @yy=" << yy << " y=" << y << ":" << s;

      // fill depth values
      //qDebug() << "fill depth values";
      QSet<GI> nowPoly;
      for(int pid=0; pid<s.size(); pid++)
      {
        // fill depth buffer here;

        //qDebug() << "hp #" << pid;

        if(s[pid].hp%2==1)
        {
          nowPoly.insert(s[pid].geo);
        }
        else
        {
          nowPoly.remove(s[pid].geo);
          continue;
        }

        float nn;
        if(pid<s.size()-1)
        {
          nn=s[pid+1].x();
        }
        else
        {
          nn=1.0;
        }
        float nowz=s[pid].z();

        int xx = ToScreenX(s[pid].x());
        if(xx<0)
        {
          nowz+=s[pid].geo->dz/size.width()*2*(-xx);
          xx=0;
        }
//            //qDebug() << "depthfragment pushed @" << xx << ", " << yy;
        for(; ToProjX(xx)<nn && xx<size.width(); xx++)
        {
//          //qDebug() << "xx=" << xx;
          DepthFragment ng;
          ng.pos.setX(ToProjX(xx));
          ng.pos.setY(y);
//          nowz=(s[pid+1].z()-s[pid].z())/(s[pid+1].x()-s[pid].x())*(ng.pos.x()-s[pid].x())+s[pid].z();
//          if(yy=size.height()/2)
//          {
//            //qDebug() << "now geo=" << s[pid].geo->name << "xx=" << xx << "nowz=" << nowz << s[pid+1].z() << s[pid].z() << s[pid+1].x() << s[pid].x();
//          }
          ng.pos.setZ(nowz);
          ng.geo=s[pid].geo;
          nowz+=ng.geo->dz/size.width()*2;
          // calculate deoth value of a fragment here
          depthBuffer.buffer[xx+yy*size.width()].chain.push_back(ng);
        }
      }
    }

    //qDebug() << "rasterization ended";
  }

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
    //qDebug() << "per sample operation";
    for(int i=0; i<size.width()*size.height(); i++)
    {
//      //qDebug() << "pixel @ (" << i%size.width() << ", " << i/size.height() << ")" << "=" << colorBuffer.buffer[i];
      qSort(depthBuffer.buffer[i].chain);
      colorBuffer.buffer[i] = ColorPixel(0.0, 0.0, 0.0, 0.0);
      for(int j=0; j<depthBuffer.buffer[i].chain.size(); j++)
      {
        colorBuffer.buffer[i] = colorBuffer.buffer[i]+depthBuffer.buffer[i].chain[j].color;
        if(colorBuffer.buffer[i].a >=0.99) // not transparent
        {
          break;
        }
      }
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

QVector3D CPURenderer::VertexShader(QVector3D &p, QVector3D &n, QVector2D &tc)
{
  QVector4D pp(p, 1.0);
  pp = projection * camera.toMatrix() * transform.toMatrix() * pp;
  if(fabs(pp.w())<1e-3) pp.setW(1e-2);

  n = transform.rotation().toRotationMatrix()*n;

  return QVector3D(pp.x()/pp.w(), pp.y()/pp.w(), pp.z()/pp.w());
}

void CPURenderer::GeometryShader(Geometry &geo)
{

}

void CPURenderer::FragmentShader(DepthFragment &frag)
{
  GI geo = frag.geo;
  frag.color = geo->ambient;

  for(int i=0; i<lights.size(); i++)
  {

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
