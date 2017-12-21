#ifndef MESH_H
#define MESH_H

#include <QVector>
#include <QVector4D>
#include <QVector3D>
#include <QVector2D>
#include <QString>
#include <QMap>
#include <QFile>
#include <QTextStream>
#include <QQueue>
#include <QDir>

#include <iostream>
#include <iomanip>
#include "cpurenderer.h"

using namespace std;

class MyMesh;

struct MyVertex : public QVector3D
{
	bool status;

	MyVertex();
	MyVertex(QVector3D);
	void operator=(QVector3D&);
};

struct MyFace
{
	QVector<int> vindex;
	QVector<int> nindex;
	QVector<int> tindex;
	QVector3D kd;
	QVector3D ks;
	bool status;
	bool paint;

	MyFace();
	void AddVertex(int vin, int nin=-1, int tin=-1);
};

struct MyMaterial
{
  QString name;
  int illum;
  QVector3D kd;
  QVector3D ka;
  QVector3D ks;
  float ns;
  float d;
  float ni;
  QString map_ka;
  QString map_kd;
  QString map_ks;
  QString map_d;
  QString map_bump;
};

struct MyMesh
{
	public:
		MyMesh();

		int AddVertex(QVector3D vert);
		int AddNormal(QVector3D norm);
		int AddTexCoord2D(QVector2D tex);
		int AddFace(MyFace face);
		void Print(ostream& stream = cout);
		void DeleteVertex(int vindex);
		void DeleteFace(int findex);
		void CollectGarbage();
		bool CheckBorderVertex(int vid);
    float MaxSize();
    float MinSize();
    void AverageNormal();

    void Render(CPURenderer* render);
    QVector3D Normal(int fid);
    void Smooth();
    void Sharp();

		QVector<MyVertex> vertices;
		QVector<QVector3D> normals;
		QVector<QVector2D> texcoords;
		QVector<MyFace> faces;
    MyMaterial mat;
		QString name;
		bool paint;
};

struct MyModel
{
	public:
		QVector<MyMesh> meshes;

		QVector3D center;
		float scale;
		QString name;
		QString folder;

		MyModel();
		MyModel(const MyModel& model);
		~MyModel();

		void Print(ostream &stream = cout);
		void Normalize();
		void Unnormalize();
		void CollectGarbage();
    void AddModel(MyModel& a);
    void AddMesh(MyMesh m);

    void Render(CPURenderer* render);
    void Smooth();
    void Sharp();
};

Q_DECLARE_METATYPE(MyModel)

bool WriteMyModel2OBJ(QString path, MyModel model);
bool WriteMyModel2DAE(QString path, MyModel model);
MyModel LoadOBJ(QString path);

#endif // MESH_H
