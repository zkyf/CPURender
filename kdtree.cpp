#include "kdtree.h"

KDTree::KDTree() : root(nullptr) { }

KDTree::KDTree(QVector<Geometry> geos)
{
  if(geos.size()<1) return;
  root = new Node();
  for(GI i=geos.begin(); i!=geos.end(); i++)
  {
    root->gl.push_back(i);
  }
  root->depth=0;
  root->Split();
}

KDTree::~KDTree()
{
  if(root)
  {
    root->~Node();
    delete root;
  }
}

KDTree::IR KDTree::Intersect(Ray ray)
{
  if(!root) return IR();
  return root->Hit(ray);
}

KDTree::Node::Node(Node* p) : left(nullptr), right(nullptr), parent(p)
{
  min = 1e20;
  max = -1e20;
  value = 0;
  depth = -1;
}

KDTree::Node::~Node()
{
  if(left)
  {
    left->~Node();
    delete left;
    left = nullptr;
  }
  if(right)
  {
    right->~Node();
    delete right;
    right = nullptr;
  }
}

void KDTree::Node::Split()
{
  int vertCount = 0;
  for(GI i=geos.begin(); i!=geos.end(); i++)
  {
    vertCount+=i->vecs.size();
  }

  if(depth>=10)
  {
    return;
  }

  // calculate deviation on x
  float xd=0.0f;
  float xmean=0.0;
  float xmi=1e20;
  float xma=-1e20;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      xmean+=j->p.x();
      if(xmi>j->p.x()) xmi=j->p.x();
      if(xma<j->p.x()) xma=j->p.x();
    }
  }
  xmean/=vertCount;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      xd+=pow(j->p.x()-xmean, 2);
    }
  }
  xd/=vertCount;

  // calculate deviation on y
  float yd=0.0f;
  float ymean=0.0;
  float ymi=1e20;
  float yma=-1e20;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      ymean+=j->p.y();
      if(ymi>j->p.y()) ymi=j->p.y();
      if(yma<j->p.y()) yma=j->p.y();
    }
  }
  ymean/=vertCount;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      yd+=pow(j->p.y()-ymean, 2);
    }
  }
  yd/=vertCount;

  // calculate deviation on z
  float zd=0.0f;
  float zmean=0.0;
  float zmi=1e20;
  float zma=-1e20;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      zmean+=j->p.z();
      if(zmi>j->p.z()) zmi=j->p.z();
      if(zma<j->p.z()) zma=j->p.z();
    }
  }
  zmean/=vertCount;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      zd+=pow(j->p.z()-zmean, 2);
    }
  }
  zd/=vertCount;

  // determine which axis to split
  axis = 0;
  float nd = xd;
  if(depth==0)
  {
    xmin = xmi;
    xmax = xma;

    ymin = ymi;
    ymax = yma;

    zmin = zmi;
    zmax = zma;
  }

  if(yd>nd)
  {
    axis = 1;
    nd=yd;
  }
  if(zd>nd)
  {
    axis = 2;
  }

  // find median and split
  left = new Node(this);
  right = new Node(this);
  bool continueSplit = true;
  switch(axis)
  {
  case 0:
    // split on x
    QVector<float> temp;
    for(QVector<GI>::iterator ii=gl.begin(); i!=gl.end(); ii++)
    {
      GI& i=*ii;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        temp.push_back(j->p.x());
      }
    }
    qSort(temp);
    value=temp[temp.size()/2];
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      bool leftUsed = false;
      bool rightUsed = false;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.x()<=value && !leftUsed)
        {
          left->gl.push_back(i);
          leftUsed = true;
        }
        else if(!rightUsed)
        {
          right->gl.push_back(i);
          rightUsed = true;
        }
        if(leftUsed && rightUsed) break;
      }
      if(left->gl.size()==gl.size() || right->gl.size()==gl.size())
      {
        // can't split
        continueSplit = false;
      }
    }
    left->xmin=xmin;
    left->xmax=value;
    left->ymin=ymin;
    left->ymax=ymax;
    left->zmin=zmin;
    left->zmax=zmax;

    right->xmin=value;
    right->xmax=xmax;
    right->ymin=ymin;
    right->ymax=ymax;
    right->zmin=zmin;
    right->zmax=zmax;
    break;

  case 1:
    // split on y
    QVector<float> temp;
    for(QVector<GI>::iterator ii=gl.begin(); i!=gl.end(); ii++)
    {
      GI& i=*ii;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        temp.push_back(j->p.y());
      }
    }
    qSort(temp);
    value=temp[temp.size()/2];
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      bool leftUsed = false;
      bool rightUsed = false;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.y()<=value && !leftUsed)
        {
          left->gl.push_back(i);
          leftUsed = true;
        }
        else if(!rightUsed)
        {
          right->gl.push_back(i);
          rightUsed = true;
        }
        if(leftUsed && rightUsed) break;
      }
      if(left->gl.size()==gl.size() || right->gl.size()==gl.size())
      {
        // can't split
        continueSplit = false;
      }
    }
    left->ymin=ymin;
    left->ymax=value;
    left->xmin=xmin;
    left->xmax=xmax;
    left->zmin=zmin;
    left->zmax=zmax;

    right->ymin=value;
    right->ymax=ymax;
    right->xmin=xmin;
    right->xmax=xmax;
    right->zmin=zmin;
    right->zmax=zmax;
    break;

  case 2:
    // split on z
    QVector<float> temp;
    for(QVector<GI>::iterator ii=gl.begin(); i!=gl.end(); ii++)
    {
      GI& i=*ii;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        temp.push_back(j->p.z());
      }
    }
    qSort(temp);
    value=temp[temp.size()/2];
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      bool leftUsed = false;
      bool rightUsed = false;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.z()<=value && !leftUsed)
        {
          left->gl.push_back(i);
          leftUsed = true;
        }
        else if(!rightUsed)
        {
          right->gl.push_back(i);
          rightUsed = true;
        }
        if(leftUsed && rightUsed) break;
      }
      if(left->gl.size()==gl.size() || right->gl.size()==gl.size())
      {
        // can't split
        continueSplit = false;
      }
    }
    left->zmin=zmin;
    left->zmax=value;
    left->ymin=ymin;
    left->ymax=ymax;
    left->xmin=xmin;
    left->xmax=xmax;

    right->zmin=value;
    right->zmax=zmax;
    right->ymin=ymin;
    right->ymax=ymax;
    right->xmin=xmin;
    right->xmax=xmax;
    break;

  default:
    // error here
    break;
  }

  left->depth=depth+1;
  right->depth=depth+1;

  if(continueSplit)
  {
    left->Split();
    right->Split();
  }
}

KDTree::IR KDTree::Node::Hit(Ray ray)
{
  const float eps = 1e-5;
  if(ray.o.x()>=xmin+eps && ray.o.x()<=xmax-eps &&
     ray.o.y()>=ymin+eps && ray.o.y()<=ymax-eps &&
     ray.o.z()>=zmin+eps && ray.o.z()<=zmax-eps)
  {
    // ray starts insides the box, so will definitely hit
    if(left!=nullptr && right!=nullptr)
    {
      KDTree::IR lr = left->Hit(ray);
      KDTree::IR rr = right->Hit(ray);
      if(lr.d<rr.d) return lr;
      else return rr;
    }
    else
    {
      // this node is a leaf
      float min = 1e20;
      GI ming;
      for(int ii=0; ii<gl.size(); ii++)
      {
        VertexInfo hp = ray.IntersectGeo(*ii);
        if(hp.valid)
        {
          float d=QVector3D::dotProduct(hp-QVector3D(ray.o), QVector3D(ray.n));
          if(d<min)
          {
            min=d;
            ming=*ii;
          }
        }
      }

      if(min<1e20)
      {
        KDTree::IR result;
        result.d=min;
        result.geo=ming;
        result.valid=true;
        return result;
      }
      else
      {
        return KDTree::IR();
      }
    }
  }
  else
  {

    if(ray.o.x()<=xmin && ray.n.x()<=0) return KDTree::IR();
    if(ray.o.x()>=xmax && ray.n.x()>=0) return KDTree::IR();

    if(ray.o.y()<=ymin && ray.n.y()<=0) return KDTree::IR();
    if(ray.o.y()>=ymax && ray.n.y()>=0) return KDTree::IR();

    if(ray.o.z()<=zmin && ray.n.z()<=0) return KDTree::IR();
    if(ray.o.z()>=zmax && ray.n.z()>=0) return KDTree::IR();

    if(fabs(ray.n.x())>eps)
    {
      // ray is not parallel to y-z plane
    }
  }
}

KDTree::IR::IR()
{
  valid = false;
}
