#ifndef SIMPLEMESH_H
#define SIMPLEMESH_H

#include <QString>
#include <QVector>
#include <QVector2D>
#include <QVector3D>

struct SimpleVertex;
struct SimpleRenderInfo;
struct SimpleMaterial;
struct SimpleFace;
struct SimpleMesh;

typedef QLinkedList<SimpleMaterial>::iterator SMTLI;
typedef QLinkedList<SimpleVertex>::iterator SVI;
typedef QLinkedList<SimpleRenderInfo>::iterator SRI;
typedef QLinkedList<SimpleFace>::iterator SFI;
typedef QLinkedList<SimpleMesh>::iterator SMI;

typedef SimpleMaterial* SMTLP;
typedef SimpleVertex* SVP;
typedef SimpleRenderInfo* SRP;
typedef SimpleFace* SFP;
typedef SimpleMesh* SMP;

struct SimpleMaterial
{
  QString text;
  QVector3D ambient;
  QVector3D diffuse;
  QVector3D speculer;
};

struct SimpleVertex
{
  QVector3D p;
};

struct SimpleRenderInfo
{
  SimpleVertex* v;
  QVector3D n;
  QVector2D tc;
};

struct SimpleFace
{
  QLinkedList<SimpleRenderInfo> vlist;
  SimpleMaterial* mtl;
};

struct SimpleMesh
{
  QLinkedList<SimpleVertex> vlib;
  QLinkedList<SimpleFace> flib;
  QLinkedList<SimpleMaterial> mlib;
};

#endif // SIMPLEMESH_H
