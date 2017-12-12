#include "cpurenderer.h"

CPURenderer::CPURenderer(QSize s) :
  size(s),
  colorBuffer(s),
  depthBuffer(s),
  stencilBuffer(nullptr)
{
  int bufferCount = size.width()*size.height();
  stencilBuffer = new uchar[bufferCount];
  memset(stencilBuffer, 0, sizeof(uchar)*bufferCount);
}

CPURenderer::~CPURenderer()
{
  delete[] stencilBuffer;
}

ColorPixel* CPURenderer::Render()
{
//  if(model==nullptr)
//  {
//    return colorBuffer;
//  }

//  geos.clear();

//  for(SFI i=model->flib.begin(); i!=model->flib.end(); i++)
//  {
//    SimpleFace& face = model->flib[i];
//    Geometry geo;
//    for(SRI j=face.vlist.begin(); j!=face.vlist.end(); j++)
//    {
//      geo.vecs.push_back(j->v->p);
//      geo.norms.push_back(j->n);
//      geo.texcoords.push_back(j->tc);
//    }
//    geos.push_back(geo);
//  }

  // vertex shader
  {
    for(GI i= geos.begin(); i!=geos.end(); i++)
    {
      for(int vi=0; vi<i->vecs.size(); vi++)
      {
        VertexShader(i->vecs[vi], i->norms[vi], i->texcoords[vi]);
      }
    }
  }

  // geometry shader
  {
    int geosNum = geos.size();
    int count = 0;
    for(GI i= geos.begin(); count<geosNum; i++)
    {
      GeometryShader(*i);
    }
  }

  // post process & clipping
  {
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      Geometry ng;
      ng.ambient=i->ambient;
      ng.diffuse=i->diffuse;
      ng.specular=i->specular;
      bool dirt = false;
      bool last = false; // true=in, false=out

      QVector3D p = i->vecs[0];
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

      for(int j=1; j<i->vecs.size(); j++)
      {
        QVector3D p = i->vecs[j];
        if(p.x()>=-1 && p.y()<=1 && p.z()<= farPlane && p.z()>=nearPlane)
        {
          ng.vecs.push_back(p);
          ng.norms.push_back(i->norms[j]);
          ng.texcoords.push_back(i->texcoords[j]);

          if(last==false)
          {
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

          ng.vecs.push_back(p);
          last=true;
        }
        else
        {
          dirt=true;
          if(last==true)
          {
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
  }

  // rasterization
  {
    qSort(geos);

    float top=-1e20, bottom=1e20;
    for(GI i=geos.begin(); i!=geos.end(); i++)
    {
      if(i->vecs.size()<3)
      {
        continue;
      }

      if(i->top>top) top=i->top;
      if(i->bottom<bottom) bottom=i->bottom;
    }
    int tt=ToScreenY(top);
    int bb=ToScreenY(bottom);

    for(int y=tt; y<bb; y++)
    {
      float yy=ToProjY(y);
      Scanline s;
      for(GI i=geos.begin(); i!=geos.end(); i++)
      {
        if(ToScreenY(i->top)<y || ToScreenY(i->bottom)>y)
        {
          continue;
        }

        int c=0;
        for(int j=0; j<i->vecs.size(); j++)
        {
          int next=(j+1)%i->vecs.size();
          if((yy-i->vecs[j].y())*(i->vecs[next]-yy)<=0)
          {
            c++;
            float r=(i->vecs[j].y()-yy)/(i->vecs[next].y()-i->vecs[j].y());
            ScanlinePoint np(r*i->vecs[next]+(1-r)*i->vecs[j]);

            np.geo=i;
            np.hp=c;
          }
        }
      }

      qSort(s);

      QSet<GI> nowPoly;
      int pid=0;
      for(int pid=0; pid<s.size(); pid++)
      {
        // fill depth buffer here;

        if(s[pid].hp%2==0)
        {
          nowPoly.remove(s[pid].geo);
        }
        else
        {
          nowPoly.insert(s[pid].geo);
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

        }
      }
    }
  }

  // per sample operations
  {
  }
}
