#include "mesh.h"
#include <QDebug>

MyVertex::MyVertex()
{
	setX(0);
	setY(0);
	setZ(0);
	status = true;
}

MyVertex::MyVertex(QVector3D b)
{
	setX(b.x());
	setY(b.y());
	setZ(b.z());
	status = true;
}

void MyVertex::operator=(QVector3D& b)
{
	setX(b.x());
	setY(b.y());
	setZ(b.z());
	status = true;
}

MyFace::MyFace()
{
	status = true;
	paint = true;
}

void MyFace::AddVertex(int vin, int nin, int tin)
{
	vindex.push_back(vin);
	nindex.push_back(nin);
	tindex.push_back(tin);
}

MyMesh::MyMesh() :
	name("mesh")
{
	paint = true;
}

int MyMesh::AddVertex(QVector3D vert)
{
	vertices.push_back(vert);
	return vertices.size();
}

int MyMesh::AddNormal(QVector3D norm)
{
	norm.normalize();
	normals.push_back(norm);
	return normals.size();
}

int MyMesh::AddTexCoord2D(QVector2D tex)
{
	texcoords.push_back(tex);
	return texcoords.size();
}

int MyMesh::AddFace(MyFace face)
{
	faces.push_back(face);
	return faces.size();
}

bool MyMesh::CheckBorderVertex(int vid)
{
	bool result = false;
	for(int i=0; i<vertices.size(); i++)
	{
		if(i==vid) continue;
		int count=0;
		for(int fid=0; fid<faces.size(); fid++)
		{
			bool hasVid = false;
			bool hasI = false;
			for(int vin=0; vin<faces[fid].vindex.size(); vin++)
			{
				if(faces[fid].vindex[vin]==vid)
				{
					hasVid=true;
				}
				if(faces[fid].vindex[vin]==i)
				{
					hasI=true;
				}
				if(hasVid&&hasI)
				{
//					qDebug() << "MyMesh::CheckBorderVertex : " << "vid=#" << vid << "i=" << i << " are both in face#" << fid;
					count++;
					break;
				}
			}
		}
//		qDebug() << "MyMesh::CheckBorderVertex : " << "vid#" << vid << " count=" << count;
		if(count==1)
		{
			result=true;
			break;
		}
	}

	return result;
}

void MyMesh::Print(ostream &stream)
{
	stream << "Vertex number: " << vertices.size() << endl;
	stream << "Normal number: " << normals.size() << endl;
	stream << "Texture coords number: " << texcoords.size() << endl;
	stream << "Face number: " << faces.size() << endl;
	stream << "Vertices: " << endl;
	for(int i=0; i<vertices.size(); i++)
	{
		stream << "  v " << vertices[i].x() << " " << vertices[i].y() << " " << vertices[i].z() << endl;
	}
	stream << "Normals: " << endl;
	for(int i=0; i<normals.size(); i++)
	{
		stream << "  vn " << normals[i].x() << " " << normals[i].y() << " " << normals[i].z() << endl;
	}
	stream << "Texcoords: " << endl;
	for(int i=0; i<texcoords.size(); i++)
	{
		stream << "  vt " << texcoords[i].x() << " " << texcoords[i].y() << endl;
	}
	stream << "Faces:" << endl;
	for(int i=0; i<faces.size(); i++)
	{
		stream << "  f ";
		MyFace face = faces[i];
		for(int j=0; j<face.vindex.size(); j++)
		{
			stream << face.vindex[j] << "/" << (face.nindex.size()>0?face.nindex[j]:-1) << "/" << (face.tindex.size()>0?face.tindex[j]:-1) << " ";
		}
		stream << endl;
	}
}

void MyMesh::DeleteVertex(int vindex)
{
	for(int i=0; i<faces.size(); i++)
	{
		for(int j=0; j<faces[i].vindex.size(); j++)
		{
			if(faces[i].vindex[j] == vindex)
			{
				faces[i].status = false;
				break;
			}
		}
	}
	vertices[vindex].status = false;
}

void MyMesh::DeleteFace(int findex)
{
	faces[findex].status = false;
}

void MyMesh::CollectGarbage()
{

	/* BUG INFO
	 * Shouldnt assume vertices, normals and texcoords are correspongingly listed
	 */

	qDebug() << "MyMesh::CollectGarbage : " << "delete faces";
	// delete faces
	for(int i=faces.size()-1; i>=0; i--)
	{
		if(faces[i].status==false)
		{
			faces.remove(i);
		}
		else
		{
			for(int j=0; j<faces[i].vindex.size(); j++)
			{
				int vid = faces[i].vindex[j];
				if(vertices[vid].status==false)
				{
					faces.remove(i);
					break;
				}
			}
		}
	}

	qDebug() << "MyMesh::CollectGarbage : " << "delete vertices";
	// delete vertices
	QVector<int> newIDs;
	int newID = 0;
	for(int i=0; i<vertices.size(); i++)
	{
		if(vertices[i].status)
		{
			newIDs.push_back(newID++);
		}
		else
		{
			newIDs.push_back(-1);
		}
	}

	for(int i=0; i<vertices.size(); i++)
	{
		qDebug() << "MyMesh::CollectGarbage : " << "oldID=" << i << "newID=" << newIDs[i];
	}
	qDebug() << "MyMesh::CollectGarbage : " << "stage 1 done";
	for(int i=vertices.size()-1; i>=0; i--)
	{
		if(vertices[i].status == false)
		{
			vertices.remove(i);
		}
	}
	qDebug() << "MyMesh::CollectGarbage : " << "stage 2 done";
	for(int i=0; i<faces.size(); i++)
	{
		qDebug() << "MyMesh::CollectGarbage : " << "fid=" << i;
		for(int j=0; j<faces[i].vindex.size(); j++)
		{
			qDebug() << "MyMesh::CollectGarbage : " << "vid=" << j;
      qDebug() << "MyMesh::CollectGarbage : " << "oldvID=" << faces[i].vindex[j] << vertices[faces[i].vindex[j]].status;
			faces[i].vindex[j] = newIDs[faces[i].vindex[j]];
			qDebug() << "MyMesh::CollectGarbage : " << "newvID=" << faces[i].vindex[j];
		}
	}
  qDebug() << "MyMesh::CollectGarbage : " << "done";
}

float MyMesh::MaxSize()
{
  float xmin=1e20, xmax=-1e20;
  float ymin=1e20, ymax=-1e20;
  float zmin=1e20, zmax=-1e20;
  return 0;
}

float MyMesh::MinSize()
{
  return 0;
}

void MyMesh::Render(CPURenderer *render)
{
  for(int fid=0; fid<faces.size(); fid++)
  {
    Geometry geo;
    bool noNorm=false;
    for(int vid=0; vid<faces[fid].vindex.size(); vid++)
    {
      VertexInfo v(vertices[faces[fid].vindex[vid]]);
      if(faces[fid].nindex.size()>vid && faces[fid].nindex[vid]>0)
      {
        v.n=normals[faces[fid].nindex[vid]];
      }
      else
      {
        noNorm=true;
      }
      if(faces[fid].tindex.size()>vid && faces[fid].tindex[vid]>0)
      {
        v.tc=texcoords[faces[fid].tindex[vid]];
      }
      geo.vecs.push_back(v);
    }
    if(noNorm) geo.SetNormal();
    geo.ambient=faces[fid].mat.ka;
    geo.diffuse=faces[fid].mat.kd;
    geo.specular=faces[fid].mat.ks;
    geo.text=faces[fid].mat.mdid;
    geo.stext=faces[fid].mat.msid;
    geo.ns=faces[fid].mat.ns;
    render->AddGeometry(geo);
  }

  for(int i=0; i<textures.size(); i++)
  {
    render->AddTexture(textures[i]);
  }
}

QVector3D MyMesh::Normal(int fid)
{
  MyFace& face=faces[fid];
  if(face.vindex.size()<3) return QVector3D(0, 0, 0);
  return QVector3D::crossProduct(vertices[face.vindex[2]]-vertices[face.vindex[1]], vertices[face.vindex[0]]-vertices[face.vindex[1]]);
}

void MyMesh::Smooth()
{
  normals.clear();
  for(int vid=0; vid<vertices.size(); vid++)
  {
    normals.push_back(QVector3D(0, 0, 0));
  }
  for(int fid=0; fid<faces.size(); fid++)
  {
    MyFace& face=faces[fid];
    QVector3D n=Normal(fid);
    face.nindex.clear();
    for(int vid=0; vid<face.vindex.size(); vid++)
    {
      normals[face.vindex[vid]]+=n;
      face.nindex.push_back(face.vindex[vid]);
    }
  }
  for(int vid=0; vid<vertices.size(); vid++)
  {
    normals[vid].normalize();
  }
}

void MyMesh::Sharp()
{
  normals.clear();
  for(int fid=0; fid<faces.size(); fid++)
  {
    MyFace& face=faces[fid];
    QVector3D n=Normal(fid);
    normals.push_back(n);
    face.nindex.clear();
    for(int vid=0; vid<face.vindex.size(); vid++)
    {
      face.nindex.push_back(fid);
    }
  }
  for(int vid=0; vid<vertices.size(); vid++)
  {
    normals[vid].normalize();
  }
}

void MyMesh::Clear()
{
  faces.clear();
  vertices.clear();
  normals.clear();
  texcoords.clear();
  name = "";
}

MyModel::MyModel()
{
	center = QVector3D(0, 0, 0);
	scale = 1.0;
}

MyModel::MyModel(const MyModel &model)
{
	meshes = model.meshes;
	center = model.center;
	scale = model.scale;
	name = model.name;
	folder = model.folder;
}

MyModel::~MyModel()
{

}

void MyModel::Print(ostream& stream)
{
	stream << "Mesh: " << name.toStdString() << endl;
	stream << "Center: (" << center.x() << ", " << center.y() << ", " << center.z() << ")" << endl;
	stream << "Scale: " << scale << endl;
	stream << "Mesh number: " << meshes.size() << endl;
	int totalFace = 0;
	for(int i=0; i<meshes.size(); i++)
	{
		totalFace+= meshes[i].faces.size();
	}
	stream << "Total faces: " << totalFace << endl;
	for(int i=0; i<meshes.size(); i++)
	{
		stream << "Mesh #" << i << endl;
		meshes[i].Print(stream);
		stream << endl;
	}
}

void MyModel::Normalize()
{
	float xmin, xmax, ymin, ymax, zmin, zmax;
	float max = -1e20;
    xmin=1e20;ymin=1e20;zmin=1e20;
    xmax=-1e20;ymax=-1e20;zmax=-1e20;

	for(int mid=0; mid<meshes.size(); mid++)
	{
		MyMesh& mesh = meshes[mid];
		for(int i=0; i<mesh.vertices.size(); i++)
		{
			if(mesh.vertices[i].x()<xmin) xmin=mesh.vertices[i].x();
			if(mesh.vertices[i].y()<ymin) ymin=mesh.vertices[i].y();
			if(mesh.vertices[i].z()<zmin) zmin=mesh.vertices[i].z();

			if(mesh.vertices[i].x()>xmax) xmax=mesh.vertices[i].x();
			if(mesh.vertices[i].y()>ymax) ymax=mesh.vertices[i].y();
			if(mesh.vertices[i].z()>zmax) zmax=mesh.vertices[i].z();
		}
		if(xmax-xmin>max) max=xmax-xmin;
		if(ymax-ymin>max) max=ymax-ymin;
		if(zmax-zmin>max) max=zmax-zmin;
	}

	center.setX((xmax+xmin)/2);
	center.setY((ymax+ymin)/2);
	center.setZ((zmax+zmin)/2);

  scale = max;

	for(int mid=0; mid<meshes.size(); mid++)
	{
		MyMesh& mesh = meshes[mid];
		for(int i=0; i<mesh.vertices.size(); i++)
		{
      mesh.vertices[i]=(mesh.vertices[i]-MyVertex(QVector3D((xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2)))/max;
		}
	}
}

void MyModel::Unnormalize()
{
	for(int mid=0; mid<meshes.size(); mid++)
	{
		MyMesh& mesh = meshes[mid];
		for(int vid=0; vid<mesh.vertices.size(); vid++)
		{
            mesh.vertices[vid] = mesh.vertices[vid]*scale+center;
		}
	}
}

void MyModel::CollectGarbage()
{
	for(int i=0; i<meshes.size(); i++)
	{
		qDebug() << "MyModel::CollectGarbage : " << "collect garbage mesh#" << i << " total " << meshes.size();
		meshes[i].CollectGarbage();
	}
  qDebug() << "MyModel:: CollectGarbage : " << "done";
}

void MyModel::AddModel(MyModel &a)
{
  for(int i=0; i<a.meshes.size(); i++)
  {
    meshes.push_back(a.meshes[i]);
  }
}

void MyModel::AddMesh(MyMesh m)
{
  meshes.push_back(m);
}

void MyModel::Render(CPURenderer *render)
{
  for(int i=0; i<meshes.size(); i++)
  {
    meshes[i].Render(render);
  }
}

void MyModel::Smooth()
{
  for(int i=0; i<meshes.size(); i++)
  {
    meshes[i].Smooth();
  }
}

void MyModel::Sharp()
{
  for(int i=0; i<meshes.size(); i++)
  {
    meshes[i].Sharp();
  }
}

bool WriteMyModel2DAE(QString path, MyModel model)
{
	model.Unnormalize();
	QString fileName = path+"/"+model.name+".dae";
	QFile outFile(fileName);
    QFileInfo fileInfo(fileName);
    if(fileInfo.exists())
    {
        QFile::remove(fileName);
    }
    if(!outFile.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		return false;
	}
	QTextStream out(&outFile);
    out.setRealNumberPrecision(16);

	out << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << endl;
	out << "<COLLADA xmlns=\"http://www.collada.org/2005/11/COLLADASchema\" version=\"1.4.1\">" << endl;
    out << "<asset>" << endl;
    out << "<created>2017-07-11T20:24:41</created>" << endl;
    out << "<modified>2017-07-11T20:24:41</modified>" << endl;
    out << "<up_axis>Y_UP</up_axis>" << endl;
    out << "</asset>" << endl;

	// output materials
	out << "  <library_materials>" << endl;
	for(int i=0; i<model.meshes.size(); i++)
	{
		QString matName = QString("material_%1").arg(i);
		QString effName = QString("effect_%1").arg(i);
		out << "    <material id=\"" << matName << "\" name=\"" << matName << "\">" << endl;
		out << "      <instance_effect url=\"#" << effName << "\" />" << endl;
		out << "    </material>" << endl;
	}
	out << "  </library_materials>" << endl;

	// output effects
	out << "  <library_effects>" << endl;
	for(int i=0; i<model.meshes.size(); i++)
	{
		QString effName=QString("effect_%1").arg(i);
		out << "    <effect id=\"" << effName << "\">" << endl;
		out << "      <profile_COMMON>" << endl;
		out << "        <newparam sid=\"" << effName+"_tex" << "\">" << endl;
		out << "          <surface type=\"2D\">" << endl;
		out << "            <init_from>image_" << i << "</init_from>" << endl;
		out << "          </surface>" << endl;
		out << "        </newparam>" << endl;
		out << "        <newparam sid=\"" << effName+"_sampler" << "\">" << endl;
		out << "          <sampler2D>" << endl;
		out << "            <source>" << effName+"_tex" << "</source>" << endl;
		out << "          </sampler2D>" << endl;
		out << "        </newparam>" << endl;
		out << "        <technique sid=\"common\">" << endl;
		out << "          <lambert>" << endl;
		out << "            <ambient>" << endl;
		out << "              <color>0 0 0 1</color>" << endl;
		out << "            </ambient>" << endl;
		out << "            <diffuse>" << endl;
		out << "              <texture texture=\"" << effName+"_sampler" << "\" texcoord=\"uvset_" << i << "\">" << endl;
		out << "              </texture>" << endl;
		out << "            </diffuse>" << endl;
		out << "          </lambert>" << endl;
		out << "        </technique>" << endl;
		out << "      </profile_COMMON>" << endl;
		out << "    </effect>" << endl;
	}
	out << "  </library_effects>" << endl;

	// output images
	out << "  <library_images>" << endl;
	for(int i=0; i<model.meshes.size(); i++)
	{
		out << "    <image id=\"image_" << i << "\">" << endl;
//		if(model.meshes[i].textures.size()!=0)
		{
//			out << "      <init_from>" << model.meshes[i].textures[0].name << "</init_from>" << endl;
		}
		out << "    </image>" << endl;
	}
	out << "  </library_images>" << endl;

	// output geometries
	out << "  <library_geometries>" << endl;
	for(int i=0; i<model.meshes.size(); i++)
	{
		MyMesh& mesh = model.meshes[i];
		QString matName = QString("material_%1").arg(i);
		QString geoID = QString("geometry_%1").arg(i);
		QString geoName = QString("geo_%1").arg(i);
		QString geoPosSourceID = geoID+"_pos";
		QString geoPosSourceName = geoName+"_pos";
		QString geoPosArray = geoID+"_array";
		out << "    <geometry id=\"" << geoID << "\" name=\"" << geoName << "\">" << endl;
		out << "		  <mesh>" << endl;
		out << "        <source id=\"" << geoPosSourceID << "\" name=\"" << geoPosSourceName << "\">" << endl;
		out << "          <float_array id=\"" << geoPosArray << "\" count=\"" << mesh.vertices.size()*3 << "\">" << endl;
		for(int vid=0; vid<mesh.vertices.size(); vid++)
		{
            out << "          " << mesh.vertices[vid].x() << " " << mesh.vertices[vid].y() << " " << mesh.vertices[vid].z() << endl;
		}
		out << "          </float_array>" << endl;
		out << "          <technique_common>" << endl;
		out << "            <accessor source=\"#" << geoPosArray << "\" count=\"" << mesh.vertices.size() << "\" stride=\"3\">" << endl;
		out << "              <param name=\"X\" type=\"float\"/>" << endl;
		out << "              <param name=\"Y\" type=\"float\"/>" << endl;
		out << "              <param name=\"Z\" type=\"float\"/>" << endl;
		out << "            </accessor>" << endl;
		out << "          </technique_common>" << endl;
		out << "        </source>" << endl;

		QString uvSourceID = QString("uv_%1").arg(i);
		QString uvSourceName = QString("uvset_%1").arg(i);
		QString uvArray = QString("uv__array_%1").arg(i);
		out << "        <source id=\"" << uvSourceID << "\" name=\"" << uvSourceName << "\">" << endl;
		out << "          <float_array id=\"" << uvArray << "\" count=\"" << mesh.texcoords.size()*2 << "\">" << endl;
		for(int tid=0; tid<mesh.texcoords.size(); tid++)
		{
			out << "          " << mesh.texcoords[tid].x() << " " << mesh.texcoords[tid].y() << endl;
		}
		out << "          </float_array>" << endl;
		out << "          <technique_common>" << endl;
		out << "            <accessor source=\"#" << uvArray << "\" count=\"" << mesh.texcoords.size() << "\" stride=\"2\">" << endl;
		out << "              <param name=\"S\" type=\"float\"/>" << endl;
		out << "              <param name=\"T\" type=\"float\"/>" << endl;
		out << "            </accessor>" << endl;
		out << "          </technique_common>" << endl;
		out << "        </source>" << endl;

		QString vertID = QString("vertices_%1").arg(i);
		out << "      <vertices id=\"" << vertID << "\">" << endl;
		out << "        <input semantic=\"POSITION\" source=\"#" << geoPosSourceID << "\"/>" << endl;
		out << "      </vertices>" << endl;
		out << "      <triangles material=\"" << matName << "\" count=\"" << mesh.faces.size() << "\">" << endl;
		out << "        <input offset=\"0\" semantic=\"VERTEX\" source=\"#" << vertID << "\"/>" << endl;
		out << "        <input offset=\"1\" semantic=\"TEXCOORD\" source=\"#" << uvSourceID << "\"/>" << endl;
		out << "      <p>" << endl;
		for(int fid=0; fid<mesh.faces.size(); fid++)
		{
			MyFace& face=mesh.faces[fid];
			for(int vid=0; vid<face.vindex.size(); vid++)
			{
				out << "      " << face.vindex[vid] << " " << face.tindex[vid] << endl;
			}
		}
		out << "      </p>" << endl;
		out << "      </triangles>" << endl;
		out << "      </mesh>" << endl;
		out << "    </geometry>" << endl;
	}
	out << "  </library_geometries>" << endl;

	// output visual scenes
	out << "  <library_visual_scenes>" << endl;
	out << "    <visual_scene id=\"visual_scene_node\" name=\"scene\">" << endl;
	for(int i=0; i<model.meshes.size(); i++)
	{
		QString matName = QString("material_%1").arg(i);
		QString geoID = QString("geometry_%1").arg(i);
		QString geoName = QString("geo_%1").arg(i);
		out << "      <node name=\"" << geoName << "\" type=\"NODE\">" << endl;
		out << "        <instance_geometry url=\"#" << geoID << "\">" << endl;
		out << "          <bind_material>" << endl;
		out << "            <technique_common>" << endl;
		out << "              <instance_material symbol=\"" << matName << "\" target=\"#" << matName << "\">" << endl;
		out << "                <bind_vertex_input semantic=\"uvset_" << i << "\" input_semantic=\"TEXCOORD\" input_set=\"0\"/>" << endl;
		out << "              </instance_material>" << endl;
		out << "            </technique_common>" << endl;
		out << "          </bind_material>" << endl;
		out << "        </instance_geometry>" << endl;
		out << "      </node>" << endl;
	}
	out << "    </visual_scene>" << endl;
	out << "  </library_visual_scenes>" << endl;
	out << "  <scene>" << endl;
	out << "    <instance_visual_scene url=\"#visual_scene_node\"/>" << endl;
	out << "  </scene>" << endl;

	out << "</COLLADA>" << endl;
}

bool WriteMyModel2OBJ(QString path, MyModel model)
{
	QFile objFile(path+"/"+model.name+".obj");
	if(!objFile.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		return false;
	}
	QTextStream objOut(&objFile);

	QFile mtlFile(path+"/"+model.name+".mtl");
	if(!mtlFile.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		return false;
	}
	QTextStream mtlOut(&mtlFile);

	objOut << "# temp obj file by BillboardCloudSimplification" << endl;
	objOut << "# By Jiaxin Liu" << endl;
	objOut << "mtllib " << model.name << ".mtl" << endl;
	objOut << endl;

	objOut << "o object" << endl;
	int nowVertiNum = 1;
	int nowTexcoNum = 1;
	int nowNormNum = 1;

	for(int meshID=0; meshID<model.meshes.size(); meshID++)
	{
		MyMesh& mesh= model.meshes[meshID];
		// output material lib
		objOut << "usemtl " << QString("mesh%1").arg(meshID) << endl;
		// output vertices
		for(int vid=0; vid<mesh.vertices.size(); vid++)
		{
			QVector3D& v=mesh.vertices[vid];
			objOut << "v " << v.x() << " " << v.y() << " " << v.z() << endl;
		}

		// output texcoords
		for(int vtid=0; vtid<mesh.texcoords.size(); vtid++)
		{
			QVector2D& vt=mesh.texcoords[vtid];
			objOut << "vt " << vt.x() << " " << vt.y() << endl;
		}

		// output faces
		for(int fid=0; fid<mesh.faces.size(); fid++)
		{
			MyFace& face = mesh.faces[fid];
			objOut << "f ";
			for(int fvid=0; fvid<face.vindex.size(); fvid++)
			{
				objOut << face.vindex[fvid]+nowVertiNum << "/";
				if(face.tindex.size()>fvid)
					objOut << face.tindex[fvid]+nowTexcoNum;
				//objOut << "/";
				if(face.nindex.size()>fvid)
				{
					//objOut << face.nindex[fvid]+nowNormNum;
				}
				objOut << " ";
			}
			objOut << endl;
		}
		objOut << endl;

		nowVertiNum += mesh.vertices.size();
		nowTexcoNum += mesh.texcoords.size();
		nowNormNum += mesh.normals.size();
	}

	// output mtl file
	for(int meshID=0; meshID<model.meshes.size(); meshID++)
	{
		MyMesh& mesh=model.meshes[meshID];
		mtlOut << "newmtl " << QString("mesh%1").arg(meshID) << endl;
//		mtlOut << "Kd " << mesh.diffuse.x() << " " << mesh.diffuse.y() << " " << mesh.diffuse.z() << endl;
//		if(model.meshes[meshID].textures.size()>0)
//			mtlOut << "map_Kd " << model.meshes[meshID].textures[0].name << endl;
	}

	objFile.close();
	mtlFile.close();
	return true;
}

MyModel LoadOBJ(QString path)
{
  QFileInfo fileInfo(path);
  if(!fileInfo.exists()) return MyModel();
  QString fileDir = fileInfo.absolutePath();
  QString fileName = fileInfo.bundleName();

  QFile file(path);
  if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
  {
    return MyModel();
  }

  QTextStream in(&file);
  MyModel result;
  result.folder = fileDir;
  MyMesh mesh;
  QVector<MyMaterial> mtllib;
  int nowMat=-1;
  MyMaterial defaultMat;
  defaultMat.ka = QVector3D(0.0, 1.0, 0.0);
  defaultMat.kd = QVector3D(0.0, 1.0, 0.0);
  defaultMat.ks = QVector3D(1.0, 1.0, 1.0);
  defaultMat.ns = 16.0;

  while(!in.atEnd())
  {
    QString c;
    in >> c;
    c=c.toUpper();
//    qDebug() << c << ":";
    if(c=="#")
    {
      in.readLine();
    }
    else if(c=="V")
    {
      float x,y,z;
      in >> x >> y >> z;
      mesh.AddVertex(QVector3D(x, y, z));
    }
    else if(c=="VT")
    {
      float x,y;
      in >> x >> y;
      mesh.AddTexCoord2D(QVector2D(x, y));
    }
    else if(c=="VN")
    {
      float x,y,z;
      in >> x >> y >> z;
      mesh.AddNormal(QVector3D(x, y, z));
    }
    else if(c=="F")
    {
      MyFace face;
      QString line = in.readLine();
//      qDebug() << "F line" << line;
      QStringList ls = line.split(' ');
      QString s;
      for(int i=1; i<ls.size(); i++)
      {
        s = ls.at(i);
        QStringList l=s.split('/');
        int v=l.at(0).toInt()-1;
        int n=-1, t=-1;
        if(l.size()>1)
        {
          t=l.at(1).toInt()-1;
        }
        if(l.size()>2)
        {
          n=l.at(2).toInt()-1;
        }
        face.vindex.push_back(v);
        face.nindex.push_back(n);
        face.tindex.push_back(t);
      }
      if(nowMat>=0) face.mat=mtllib[nowMat];
      else
      {
        face.mat=defaultMat;
      }
      mesh.AddFace(face);
    }
    else if(c=="MTLLIB")
    {
      QString mtlname;
      in >> mtlname;
      mtllib = LoadMTL(fileDir+"/"+mtlname);

      for(int i=0; i<mtllib.size(); i++)
      {
        qDebug() << "mat #" << i << "map_kd=" << mtllib[i].map_kd;
        if(mtllib[i].map_kd!="")
        {
          mesh.textures.push_back(QImage(fileDir+"/"+mtllib[i].map_kd));
          mtllib[i].mdid=mesh.textures.size()-1;
        }
        if(mtllib[i].map_ks!="")
        {
          mesh.textures.push_back(QImage(fileDir+"/"+mtllib[i].map_ks));
          mtllib[i].msid=mesh.textures.size()-1;
        }
      }
    }
    else if(c=="USEMTL")
    {
      QString mat;
      in >> mat;
      for(int i=0; i<mtllib.size(); i++)
      {
        if(mtllib[i].name==mat)
        {
          nowMat=i;
          break;
        }
      }
    }
  }
  if(mesh.faces.size()!=0)
  {
    result.AddMesh(mesh);
  }
  return result;
}

QVector<MyMaterial> LoadMTL(QString path)
{
  qDebug() << "LoadMTL: " << path;
  QFileInfo fileInfo(path);
  QFile file(path);
  if(!file.open(QIODevice::Text | QIODevice::ReadOnly))
  {
    return QVector<MyMaterial>();
  }
  QTextStream in(&file);

  QVector<MyMaterial> result;
  bool dirty=false;
  MyMaterial mat;
  while(!in.atEnd())
  {
    QString c;
    in >> c;
    c=c.toUpper();
//    qDebug() << c << ":";
    if(c=="#")
    {
      in.readLine();
    }
    else if(c=="NEWMTL")
    {
      if(dirty)
      {
        result.push_back(mat);
        mat = MyMaterial();
      }

      in >> mat.name;
      dirty=false;
    }
    else if(c=="KA")
    {
      float r, g, b;
      in >> r >> g >> b;
      mat.ka = QVector3D(r, g, b);
      dirty=true;
    }
    else if(c=="KD")
    {
      float r, g, b;
      in >> r >> g >> b;
      mat.kd = QVector3D(r, g, b);
      dirty=true;
    }
    else if(c=="KS")
    {
      float r, g, b;
      in >> r >> g >> b;
      mat.ks = QVector3D(r, g, b);
      dirty=true;
    }
    else if(c=="NS")
    {
      in >> mat.ns;
      dirty=true;
    }
    else if(c=="MAP_KA")
    {
      dirty=true;
      in >> mat.map_ka;
    }
    else if(c=="MAP_KD")
    {
      dirty=true;
      in >> mat.map_kd;
    }
    else if(c=="MAP_KS")
    {
      dirty=true;
      in >> mat.map_ks;
    }
  }

  if(dirty)
  {
    result.push_back(mat);
  }
  return result;
}
