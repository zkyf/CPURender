#include "daemodel.hpp"

DaeItem::DaeItem()
{
  name = "";
  valid = false;
}

DaeMeshNode::DaeMeshNode() : DaeItem()
{
  name="";
  matrix.setToIdentity();
}

DaeMesh::DaeMesh() : DaeItem()
{
  name = "mesh";
  material = "";
  index = -1;
}

DaeNode::DaeNode() : DaeItem()
{
  name = "";
  matrix.setToIdentity();
  scale = QVector3D(1, 1, 1);
  meshNumber = 0;
}

int DaeNode::CountMeshes()
{
  meshNumber = 0;
  meshNumber+=meshes.size();
  for(int i=0; i<nodes.size(); i++)
  {
    meshNumber+=nodes[i].CountMeshes();
  }
}

DaeMeshNodeInfo DaeNode::GetMeshNode(int index)
{
  if(index<0)
  {
    return DaeMeshNodeInfo();
  }
  if(index<meshes.size())
  {
    DaeMeshNodeInfo result;
    result.matrix=matrix;
    result.name=meshes[i].name;
    result.material=meshes[i].material;
  }
  else
  {
    int currentCap=meshes.size();
    int lastCap=meshes.size();
    for(int i=0; i<nodes.size(); i++)
    {
      currentCap+=nodes[i].meshNumber;
      if(current>index)
      {
        DaeMeshNodeInfo result=nodes[i].GetMeshNode(index-lastCap);
        result.matrix*=matrix;
        result.valid=true;
        return result;
      }
      lastCap=currentCap;
    }
  }

  return DaeMeshNodeInfo();
}

DaeMaterial::DaeMaterial() : DaeItem()
{
  name = "";
  effect = "";
}

DaeEffect::DaeEffect() : DaeItem()
{
  name = "";
  image = "";
}

DaeImage::DaeImage() : DaeItem()
{
  name = "";
  filename = "";
}

bool DaeFace::hasVertices()
{
  return !vindex.empty();
}

bool DaeFace::hasNormals()
{
  return !nindex.empty();
}

bool DaeFace::hasTexcoords()
{
  return !tindex.empty();
}

bool DaeGeometry::hasVertices()
{
  return !vertices.empty();
}

bool DaeGeometry::hasNormals()
{
  return !normals.empty();
}

bool DaeGeometry::hasTexcoords()
{
  return !texcoords.empty();
}

DaeGeometry& DaeModel::GetGeometry(QString id)
{
  for(int i=0; i<geometries.size(); i++)
  {
    if(geometries[i].name==id)
    {
      return geometries[i];
    }
  }
  return DaeGeometry();
}

DaeImage& DaeModel::GetImage(QString id)
{
  for(int i=0; i<images.size(); i++)
  {
    if(images[i].name==id)
    {
      return images[i];
    }
  }
  return DaeImage();
}

DaeEffect& DaeModel::GetEffect(QString id)
{
  for(int i=0; i<effects.size(); i++)
  {
    if(effects[i].name==id)
    {
      return effects[i];
    }
  }
  return DaeEffect();
}

DaeMaterial& DaeModel::GetMaterial(QString id)
{
  for(int i=0; i<materials.size(); i++)
  {
    if(materials[i].name==id)
    {
      return materials[i];
    }
  }
  return DaeMaterial();
}

DaeMeshNode DaeVisualScene::GetMesh(int id)
{
  int currentCap=0;
  int lastCap=0;
  for(int i=0; i<nodes.size(); i++)
  {
    currentCap+=nodes[i].meshNumber;
    if(currentCap>id)
    {
      return nodes[i].GetMesh(id-lastCap);
    }
    lastCap=currentCap;
  }
  return DaeMeshNode();
}

int DaeVisualScene::CountMeshes()
{
  meshNumber=0;
  for(int i=0; i<nodes.size(); i++)
  {
    meshNumber+=nodes[i].CountMeshes();
  }
}

DaeMeshNode DaeModel::GetMeshNode(int index, int sceneId)
{
  if(sceneId<0 || sceneId>=visual_scenes.size())
  {
    return DaeGeometry();
  }
  DaeMeshNodeInfo meshNodeInfo=visual_scenes[sceneId].GetMesh(index);
  DaeMeshNode result;
  result.valid=meshNodeInfo.valid;
  if(!result.valid)
  {
    return result;
  }
  result.matrix=meshNodeInfo.matrix;
  result.geometry=GetGeometry(meshNodeInfo.name);
  result.material=GetMaterial(meshNodeInfo.material);
  result.effect=GetEffect(result.material.effect);
  result.image=GetImage(result.effect.image);

  return result;
}

int DaeModel::CountMeshes()
{
  meshNumber=0;
  for(int i=0; i<visual_scenes.size(); i++)
  {
    meshNumber+=visual_scenes[i].CountMeshes();
  }
  return meshNumber;
}

void DaeModel::Normalize()
{
  CountMeshes();
  float xmin=1e20, xmax=-1e20;
  float ymin=1e20, ymax=-1e20;
  float zmin=1e20, zmax=-1e20;
  for(int vsid=0; vsid<visual_scenes.size(); vsid++)
  {
    for(int mid=0; mid<visual_scenes[vsid].meshNumber; mid++)
    {
      DaeMeshNode meshNode = visual_scenes[vsid].GetMesh(mid);
      if(!meshNode.valid)
      {
        continue;
      }
      for(int vid=0; vid<meshNode.geometry.vertices.size(); vid++)
      {
        QVector4D vert(meshNode.geometry.vertices[vid], 1.0);
        vert=meshNode.matrix*vert;
        if(vert.x()<xmin)
        {
          xmin=vert.x();
        }
        if(vert.x()>xmax)
        {
          xmax=vert.x();
        }
        if(vert.y()<ymin)
        {
          ymin=vert.y();
        }
        if(vert.y()>ymax)
        {
          ymax=vert.y();
        }
        if(vert.z()<zmin)
        {
          zmin=vert.z();
        }
        if(vert.z()>zmax)
        {
          zmax=vert.z();
        }
      }
    }
  }

  float xmid=(xmax+xmin)/2;
  float ymid=(ymax+ymin)/2;
  float zmid=(zmax+zmin)/2;

  float xsize=xmax-xmin;
  float ysize=ymax-ymin;
  float zsize=zmax-zmin;

  float size=xsize;
  if(size<ysize) size=yszie;
  if(size<zsize) size=zsize;

  QMatrix4x4 move;
  move.translate(-xmid, -ymid, -zmid);
  move.scale(1.0/size, 1.0/size, 1.0/size);

  for(int vsid=0; vsid<visual_scenes.size(); vsid++)
  {
    for(int nid=0; nid<visual_scenes[vsid].nodes.size(); nid++)
    {
      visual_scenes[vsid].nodes[nid].matrix=move*visual_scenes[vsid].nodes[nid].matrix;
    }
  }
}

void DaeMeshNode::ApplyMatrix()
{
  for(int i=0; i<geometry.vertices.size(); i++)
  {
    geometry.vertices[i]=matrix*geometry.vertices[i];
  }
  matrix.setToIdentity();
}
