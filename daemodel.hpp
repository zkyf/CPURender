#ifndef DAEMODEL_HPP
#define DAEMODEL_HPP

#include <QString>
#include <QDebug>
#include <QQueue>
#include <QTextStream>
#include <QMatrix4x4>
#include <QVector3D>
#include <QVector2D>
#include <QVector4D>
#include <QVector>
#include <iostream>
#include <fstream>

using namespace std;

struct DaeItem
{
  QString name;
  bool valid;

  DaeItem();
};

struct DaeMeshNodeInfo : public DaeItem
{
  QString name;
  QMatrix4x4 matrix;
  QString material;
  QVector3D scale;

  DaeMeshNodeInfo();
};

struct DaeMeshNode : public DaeItem
{
  QMatrix4x4 matrix;
  DaeGeometry geometry;
  DaeMaterial material;
  DaeEffect effect;
  DaeImage image;

  void ApplyMatrix();
};

struct DaeMesh : public DaeItem
{
  QString name;
  QString material;
  int index;

  DaeMesh();
};

struct DaeNode : public DaeItem
{
  QString name;
  QMatrix4x4 matrix;
  QVector3D scale;
  int meshNumber;
  QVector<DaeMesh> meshes;
  QVector<DaeNode> nodes;

  DaeNode();
  int CountMeshes();
  DaeMeshNodeInfo GetMeshNode(int index);
};

struct DaeMaterial : public DaeItem
{
  QString name;
  QString effect;

  DaeMaterial();
};

struct DaeEffect : public DaeItem
{
  QString name;
  QString image;

  DaeEffect();
};

struct DaeImage : public DaeItem
{
  QString name;
  QString filename;

  DaeImage();
};

struct DaeFace : public DaeItem
{
  QVector<int> vindex;
  QVector<int> nindex;
  QVector<int> tindex;

  bool hasVertices();
  bool hasNormals();
  bool hasTexcoords();
};

struct DaeGeometry : public DaeItem
{
  QString name;
  QVector<QVector3D> vertices;
  QVector<QVector3D> normals;
  QVector<QVector2D> texcoords;
  QVector<DaeFace> faces;

  bool hasVertices();
  bool hasNormals();
  bool hasTexcoords();
};

struct DaeVisualScene : public DaeItem
{
  QString name;
  QVector<DaeNode> nodes;
  int meshNumber;

  DaeMeshNodeInfo GetMesh(int id);
  int CountMeshes();
};

struct DaeModel : public DaeItem
{
  int meshNumber;
  QVector<DaeGeometry> geometries;
  QVector<DaeImage> images;
  QVector<DaeEffect> effects;
  QVector<DaeMaterial> materials;
  QVector<DaeVisualScene> visual_scenes;

  DaeGeometry& GetGeometry(QString id);
  DaeImage& GetImage(QString id);
  DaeEffect& GetEffect(QString id);
  DaeMaterial& GetMaterial(QString id);

  DaeMeshNode GetMeshNode(int index, int sceneId=0);
  int CountMeshes();
  void Normalize();
};

#endif // DAEMODEL_HPP
