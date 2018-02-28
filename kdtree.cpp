#include "kdtree.h"

const float eps = 1e-4;

KDTree::KDTree() : root(nullptr) { }

KDTree::KDTree(GeoList& geos)
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

void KDTree::SetTree(GeoList &geos)
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

KDTree::IR KDTree::Intersect(Ray ray, bool debuginfo)
{
  if(!root) return IR(false);
  return root->Hit(ray, debuginfo);
}

KDTree::Node::Node(Node* p) : left(nullptr), right(nullptr), parent(p)
{
  xmin = 1e20;
  xmax = -1e20;
  ymin = 1e20;
  ymax = -1e20;
  zmin = 1e20;
  zmax = -1e20;
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
  qDebug() << "";
  qDebug() << "Split @" << depth << "gl.size=" << gl.size();
  int vertCount = 0;

  for(int i=0; i<gl.size(); i++)
  {
    vertCount+=gl[i]->vecs.size();
  }

  if(depth==0)
  {
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i = *ii;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
//        if(depth>0)
//        {
//          if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
//             j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
//             j->p.z()<zmin-eps || j->p.z()>zmax+eps)
//          {
//            continue;
//          }
//        }

        if(xmin>j->p.x()) xmin=j->p.x();
        if(xmax<j->p.x()) xmax=j->p.x();

        if(ymin>j->p.y()) ymin=j->p.y();
        if(ymax<j->p.y()) ymax=j->p.y();

        if(zmin>j->p.z()) zmin=j->p.z();
        if(zmax<j->p.z()) zmax=j->p.z();
      }
    }
  }

  if(gl.size()<=1)
  {
    qDebug() << "leaf node";
    return;
  }
  else
  {
    qDebug() << "x: " << xmin << xmax << " y:" << ymin << ymax << " z:" << zmin << zmax;
  }

  // calculate deviation on x
  float xd=0.0f;
  float xmean=0.0;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
         j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
         j->p.z()<zmin-eps || j->p.z()>zmax+eps)
      {
        continue;
      }
      xmean+=j->p.x();
    }
  }
  xmean/=vertCount;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
         j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
         j->p.z()<zmin-eps || j->p.z()>zmax+eps)
      {
        continue;
      }
      xd+=pow(j->p.x()-xmean, 2);
    }
  }
  xd/=vertCount;

  // calculate deviation on y
  float yd=0.0f;
  float ymean=0.0;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
         j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
         j->p.z()<zmin-eps || j->p.z()>zmax+eps)
      {
        continue;
      }
      ymean+=j->p.y();
    }
  }
  ymean/=vertCount;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
         j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
         j->p.z()<zmin-eps || j->p.z()>zmax+eps)
      {
        continue;
      }
      yd+=pow(j->p.y()-ymean, 2);
    }
  }
  yd/=vertCount;

  // calculate deviation on z
  float zd=0.0f;
  float zmean=0.0;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
         j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
         j->p.z()<zmin-eps || j->p.z()>zmax+eps)
      {
        continue;
      }
      zmean+=j->p.z();
    }
  }
  zmean/=vertCount;
  for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
  {
    GI& i = *ii;
    for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
    {
      if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
         j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
         j->p.z()<zmin-eps || j->p.z()>zmax+eps)
      {
        continue;
      }
      zd+=pow(j->p.z()-zmean, 2);
    }
  }
  zd/=vertCount;

  // determine which axis to split
  axis = 0;
  float nd = xd;

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
  bool continueSplit = true;
  QVector<float> temp;
  switch(axis)
  {
  case 0:
    // split on x
    temp.clear();
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
           j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
           j->p.z()<zmin-eps || j->p.z()>zmax+eps)
        {
          continue;
        }
        temp.push_back(j->p.x());
      }
    }
    qSort(temp);
    qDebug() << "temp : " << temp;

    if(fabs(temp[0]-temp.last())<eps)
    {
      // can't split
      continueSplit = false;
      break;
    }
    else
    {
      left = new Node(this);
      right = new Node(this);
    }

    for(int i=temp.size()-1; i>0; i--)
    {
      if(fabs(temp[i]-temp[i-1])<eps)
      {
        temp.remove(i);
      }
    }
    qDebug() << "temp : " << temp;

    if(temp.size()%2==1)
    {
      value=temp[temp.size()/2];
    }
    else
    {
      value=(temp[temp.size()/2]+temp[temp.size()/2-1])/2;
    }
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      bool leftUsed = false;
      bool rightUsed = false;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.x()<=value && !leftUsed)
        {
//          qDebug() << "geo " << i->name << "in left @ " << j->p;
          left->gl.push_back(i);
          leftUsed = true;
        }
        else if(j->p.x()>=value && !rightUsed)
        {
//          qDebug() << "geo " << i->name << "in right @ " << j->p;
          right->gl.push_back(i);
          rightUsed = true;
        }
        if(leftUsed && rightUsed) break;
      }
    }

    if(continueSplit)
    {
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
    }

//    qDebug() << "left  -- x: " << left->xmin << left->xmax << " y:" << left->ymin << left->ymax << " z:" << left->zmin << left->zmax;
//    qDebug() << "right -- x: " << right->xmin << right->xmax << " y:" << right->ymin << right->ymax << " z:" << right->zmin << right->zmax;
    break;

  case 1:
    // split on y
    temp.clear();
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
           j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
           j->p.z()<zmin-eps || j->p.z()>zmax+eps)
        {
          continue;
        }
        temp.push_back(j->p.y());
      }
    }
    qSort(temp);
//    qDebug() << "temp : " << temp;

    if(fabs(temp[0]-temp.last())<eps)
    {
      // can't split
      continueSplit = false;
      break;
    }
    else
    {
      left = new Node(this);
      right = new Node(this);
    }

    for(int i=temp.size()-1; i>0; i--)
    {
      if(fabs(temp[i]-temp[i-1])<eps)
      {
        temp.remove(i);
      }
    }
    if(temp.size()%2==1)
    {
      value=temp[temp.size()/2];
    }
    else
    {
      value=(temp[temp.size()/2]+temp[temp.size()/2-1])/2;
    }
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      bool leftUsed = false;
      bool rightUsed = false;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.y()<=value && !leftUsed)
        {
//          qDebug() << "geo " << i->name << "in left @ " << j->p;
          left->gl.push_back(i);
          leftUsed = true;
        }
        else if(j->p.y()>=value && !rightUsed)
        {
//          qDebug() << "geo " << i->name << "in right @ " << j->p;
          right->gl.push_back(i);
          rightUsed = true;
        }
        if(leftUsed && rightUsed) break;
      }
    }

    if(continueSplit)
    {
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
    }

//    qDebug() << "left  -- x: " << left->xmin << left->xmax << " y:" << left->ymin << left->ymax << " z:" << left->zmin << left->zmax;
//    qDebug() << "right -- x: " << right->xmin << right->xmax << " y:" << right->ymin << right->ymax << " z:" << right->zmin << right->zmax;
    break;

  case 2:
    // split on z
    temp.clear();
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.x()<xmin-eps || j->p.x()>xmax+eps ||
           j->p.y()<ymin-eps || j->p.y()>ymax+eps ||
           j->p.z()<zmin-eps || j->p.z()>zmax+eps)
        {
          continue;
        }
        temp.push_back(j->p.z());
      }
    }
    qSort(temp);

    if(fabs(temp[0]-temp.last())<eps)
    {
      // can't split
      continueSplit = false;
      break;
    }
    else
    {
      left = new Node(this);
      right = new Node(this);
    }

    for(int i=temp.size()-1; i>0; i--)
    {
      if(fabs(temp[i]-temp[i-1])<eps)
      {
        temp.remove(i);
      }
    }
//    qDebug() << "temp : " << temp;
    if(temp.size()%2==1)
    {
      value=temp[temp.size()/2];
    }
    else
    {
      value=(temp[temp.size()/2]+temp[temp.size()/2-1])/2;
    }
    for(QVector<GI>::iterator ii=gl.begin(); ii!=gl.end(); ii++)
    {
      GI& i=*ii;
      bool leftUsed = false;
      bool rightUsed = false;
      for(VI j=i->vecs.begin(); j!=i->vecs.end(); j++)
      {
        if(j->p.z()<=value && !leftUsed)
        {
//          qDebug() << "geo " << i->name << "in left @ " << j->p;
          left->gl.push_back(i);
          leftUsed = true;
        }
        else if(j->p.z()>=value && !rightUsed)
        {
//          qDebug() << "geo " << i->name << "in right @ " << j->p;
          right->gl.push_back(i);
          rightUsed = true;
        }
        if(leftUsed && rightUsed) break;
      }
    }

    if(continueSplit)
    {
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
    }

//    qDebug() << "left  -- x: " << left->xmin << left->xmax << " y:" << left->ymin << left->ymax << " z:" << left->zmin << left->zmax;
//    qDebug() << "right -- x: " << right->xmin << right->xmax << " y:" << right->ymin << right->ymax << " z:" << right->zmin << right->zmax;
    break;

  default:
    // error here
    break;
  }
  qDebug() << "axis: " << axis << "  value:" << value;

  if(continueSplit)
  {
    left->depth=depth+1;
    right->depth=depth+1;
    left->Split();
    right->Split();
  }
}

KDTree::IR KDTree::Node::Hit(Ray ray, bool debuginfo)
{
  if(debuginfo)
  {
    qDebug() << "";
    qDebug() << "Hit :" << ray;
    qDebug() << "Node axis=" << (int)axis;
    qDebug() << "x:" << xmin << xmax << ", y:" << ymin << ymax << ", z:" << zmin << zmax;
  }

  if(ray.o.x()>=xmin+eps && ray.o.x()<=xmax-eps &&
     ray.o.y()>=ymin+eps && ray.o.y()<=ymax-eps &&
     ray.o.z()>=zmin+eps && ray.o.z()<=zmax-eps)
  {
    // ray starts insides the box, so will definitely hit
    KDTree::IR ir = HitChild(ray, debuginfo);
    if(debuginfo) qDebug() << "Hit got return value=" << ir.valid;
    return ir;
  }
  else
  {

    if(ray.o.x()<xmin-eps && ray.n.x()<0)
    {
      if(debuginfo) qDebug() << "<xmin";
      return KDTree::IR(false);
    }
    if(ray.o.x()>xmax+eps && ray.n.x()>0)
    {
      if(debuginfo) qDebug() << ">xmax";
      return KDTree::IR(false);
    }

    if(ray.o.y()<ymin-eps && ray.n.y()<0)
    {
      if(debuginfo) qDebug() << "<ymin";
      return KDTree::IR(false);
    }
    if(ray.o.y()>ymax+eps && ray.n.y()>0)
    {
      if(debuginfo) qDebug() << ">ymax";
      return KDTree::IR(false);
    }

    if(ray.o.z()<zmin-eps && ray.n.z()<0)
    {
      if(debuginfo) qDebug() << "<zmin";
      return KDTree::IR(false);
    }
    if(ray.o.z()>zmax+eps && ray.n.z()>0)
    {
      if(debuginfo) qDebug() << ">zmax";
      return KDTree::IR(false);
    }


    if(fabs(ray.n.x())>=fabs(ray.n.y()) && fabs(ray.n.x())>=fabs(ray.n.z()))
    {
      // ray is not parallel to y-z plane
      QVector4D h_xmin = (xmin-ray.o.x())/ray.n.x()*ray.n + ray.o;
      QVector4D h_xmax = (xmax-ray.o.x())/ray.n.x()*ray.n + ray.o;

      if(debuginfo) qDebug() << "h_xmin=" << h_xmin << " h_xmax=" << h_xmax;
      if( ((ymin<=h_xmax.y()+eps && ymax>=h_xmin.y()-eps) || (ymin<=h_xmin.y()+eps && ymax>=h_xmax.y()-eps) ) &&
          ((zmin<=h_xmax.z()+eps && zmax>=h_xmin.z()-eps) || (zmin<=h_xmin.z()+eps && zmax>=h_xmax.z()-eps) ))
      {
        // intersect!
        KDTree::IR ir = HitChild(ray, debuginfo);
        if(debuginfo) qDebug() << "Hit got return value=" << ir.valid;
        return ir;
      }
      else
      {
        if(debuginfo) qDebug() << "out of range"
                               << ((ymin<=h_xmax.y()+eps && ymax>=h_xmin.y()-eps) || (ymin<=h_xmin.y()+eps && ymax>=h_xmax.y()-eps))
                               << ((zmin<=h_xmax.z()+eps && zmax>=h_xmin.z()-eps) || (zmin<=h_xmin.z()+eps && zmax>=h_xmax.z()-eps));
        return KDTree::IR(false);
      }
    }
    else if(fabs(ray.n.y())>=fabs(ray.n.x()) && fabs(ray.n.y())>=fabs(ray.n.z()))
    {
      // ray is not parallel to x-z plane
      QVector4D h_ymin = (ymin-ray.o.y())/ray.n.y()*ray.n + ray.o;
      QVector4D h_ymax = (ymax-ray.o.y())/ray.n.y()*ray.n + ray.o;
      if(debuginfo) qDebug() << "h_ymin=" << h_ymin << " h_ymax=" << h_ymax;
      if( ((xmin<=h_ymax.x()+eps && xmax>=h_ymin.x()-eps) || (xmin<=h_ymin.x()+eps && xmax>=h_ymax.x()-eps) ) &&
          ((zmin<=h_ymax.z()+eps && zmax>=h_ymin.z()-eps) || (zmin<=h_ymin.z()+eps && zmax>=h_ymax.z()-eps) ))
      {
        // intersect!
        KDTree::IR ir = HitChild(ray, debuginfo);
        if(debuginfo) qDebug() << "Hit got return value=" << ir.valid;
        return ir;
      }
      else
      {
        if(debuginfo) qDebug() << "out of range"
                               << ((xmin<=h_ymax.x()+eps && xmax>=h_ymin.x()-eps) || (xmin<=h_ymin.x()+eps && xmax>=h_ymax.x()-eps))
                               << ((zmin<=h_ymax.z()+eps && zmax>=h_ymin.z()-eps) || (zmin<=h_ymin.z()+eps && zmax>=h_ymax.z()-eps));
        return KDTree::IR(false);
      }
    }
    else
    {
      // ray is not parallel to x-y plane
      QVector4D h_zmin = (zmin-ray.o.z())/ray.n.z()*ray.n + ray.o;
      QVector4D h_zmax = (zmax-ray.o.z())/ray.n.z()*ray.n + ray.o;
      if(debuginfo) qDebug() << "h_zmin=" << h_zmin << " h_zmax=" << h_zmax;
      if( ((xmin<=h_zmax.x()+eps && xmax>=h_zmin.x()-eps) || (xmin<=h_zmin.x()+eps && xmax>=h_zmax.x()-eps) ) &&
          ((ymin<=h_zmax.y()+eps && ymax>=h_zmin.y()-eps) || (ymin<=h_zmin.y()+eps && ymax>=h_zmax.y()-eps) ))
      {
        // intersect!
        KDTree::IR ir = HitChild(ray, debuginfo);
        if(debuginfo) qDebug() << "Hit got return value=" << ir.valid;
        return ir;
      }
      else
      {
        if(debuginfo) qDebug() << "out of range"
                               << ((xmin<=h_zmax.x()+eps && xmax>=h_zmin.x()-eps) || (xmin<=h_zmin.x()+eps && xmax>=h_zmax.x()-eps))
                               << ((ymin<=h_zmax.y()+eps && ymax>=h_zmin.y()-eps) || (ymin<=h_zmin.y()+eps && ymax>=h_zmax.y()-eps));
        return KDTree::IR(false);
      }
    }
  }
}

KDTree::IR KDTree::Node::HitChild(Ray ray, bool debuginfo)
{
  if(debuginfo)
  {
    qDebug() << "";
    qDebug() << "HitChild :";
    qDebug() << "Node axis=" << (int)axis;
    qDebug() << "x:" << xmin << xmax << ", y:" << ymin << ymax << ", z:" << zmin << zmax;
  }

  if(left!=nullptr && right!=nullptr)
  {
    KDTree::IR lr = left->Hit(ray, debuginfo);
    KDTree::IR rr = right->Hit(ray, debuginfo);
    if(debuginfo) qDebug() << "HitChild got Hit results: " << "lr=" << lr.valid << "rr=" << rr.valid;
    if((lr.valid && !rr.valid) || (lr.valid && rr.valid && lr.d<rr.d)) return lr;
    else return rr;
  }
  else
  {
    // this node is a leaf
    float min = 1e20;
    GI ming;
    VertexInfo minhp;
    for(int ii=0; ii<gl.size(); ii++)
    {
      VertexInfo hp = ray.IntersectGeo(gl[ii], debuginfo);
      if(debuginfo) qDebug() << "geo #"<< ii << " : " << gl[ii]->name << " @ " << hp.valid << hp.p;
      if(hp.valid)
      {
        float d=QVector3D::dotProduct(hp.p-QVector3D(ray.o), QVector3D(ray.n));
        if(debuginfo) qDebug() << "d=" << d;
        if(d<min && d>1e-3)
        {
          min=d;
          ming=gl[ii];
          minhp = hp;

          if(debuginfo) qDebug() << "min=" << min;
        }
      }
    }

    if(min<1e20)
    {
      KDTree::IR result;
      result.d=min;
      result.geo=ming;
      result.hp=minhp;
      result.valid=true;

      if(debuginfo) qDebug() << "valid hit";
      return result;
    }
    else
    {
      return KDTree::IR(false);
    }
  }
}

void KDTree::Node::Print()
{
  if(left && right)
  {
    left->Print();
    right->Print();
  }
  else
  {
    qDebug() << "";
    qDebug() << "Node axis=" << (int)axis << "depth=" << depth;
    qDebug() << "x:" << xmin << xmax << ", y:" << ymin << ymax << ", z:" << zmin << zmax;
    qDebug() << "leaf node" << "#=" << gl.size();
//    for(int i=0; i<gl.size(); i++)
//    {
//      qDebug() << "geo #"<< i << ": " << gl[i]->name;
//    }
  }
}

KDTree::IR::IR(bool v)
{
  valid = v;
}

void KDTree::Print()
{
  qDebug() << "KDTree:";
  if(root)
  {
    root->Print();
  }
}
