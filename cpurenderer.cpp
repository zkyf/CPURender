#include "cpurenderer.h"

CPURenderer::CPURenderer(QSize s) :
  size(s),
  colorBuffer(s),
  depthBuffer(s),
  stencilBuffer(nullptr)
{
  int bufferCount = size.width()*size.height();
  stencilBuffer = new uchar[bufferCount];
  memset(stencil, 0, sizeof(uchar)*bufferCount);
}

CPURenderer::~CPURenderer()
{
  delete[] colorBuffer;
  delete[] depthBuffer;
  delete[] stencilBuffer;
}

ColorPixel* CPURenderer::RenderSimpleMesh(SimpleMesh *model)
{
  if(model==nullptr)
  {
    return colorBuffer;
  }

  geos.clear();

  for(SFI i=model->flib.begin(); i!=model->flib.end(); i++)
  {
    SimpleFace& face = model->flib[i];
    Geometry geo;
    for(SRI j=face.vlist.begin(); j!=face.vlist.end(); j++)
    {
      geo.vecs.push_back(j->v->p);
      geo.norms.push_back(j->n);
      geo.texcoords.push_back(j->tc);
    }
    geos.push_back(geo);
  }

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
  }

  // fragment shader
  {
  }

  // per sample operations
  {
  }
}
