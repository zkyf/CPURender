#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);
  render.Resize(ui->graphicsView->size());
  render.CamTranslate(QVector3D(0, 0, 2));
  Light light;
  light.SetPointLight(QVector3D(0, 0, 2));
  render.AddLight(light);
}

MainWindow::~MainWindow()
{
  delete ui;
}

void MainWindow::on_graphicsView_KeyReleaseEvent(QKeyEvent *event)
{
//  qDebug() << "key pressed : " << event->text();

  if(event->key()==Qt::Key_A)
  {
    render.CamRotate(render.CamUp(), 1);
    NewFrame();
  }
  if(event->key()==Qt::Key_D)
  {
    render.CamRotate(render.CamUp(), -1);
    NewFrame();
  }
  if(event->key()==Qt::Key_W)
  {
    render.CamRotate(render.CamRight(), 1);
    NewFrame();
  }
  if(event->key()==Qt::Key_S)
  {
    render.CamRotate(render.CamRight(), -1);
    NewFrame();
  }

  const float tr = 0.05;
  if(event->key()==Qt::Key_J)
  {
    render.CamTranslate(-render.CamRight()*tr);
    NewFrame();
  }
  if(event->key()==Qt::Key_L)
  {
    render.CamTranslate(render.CamRight()*tr);
    NewFrame();
  }
  if(event->key()==Qt::Key_I)
  {
    render.CamTranslate(render.CamForward()*tr);
    NewFrame();
  }
  if(event->key()==Qt::Key_K)
  {
    render.CamTranslate(render.CamForward()*tr);
    NewFrame();
  }

  if(event->key()==Qt::Key_Space)
  {
    render.WorldRotate(QVector3D(0, 1, 0), 3);
    NewFrame();
  }

//  qDebug() << "now rotation: " << render.CamRotation().toEulerAngles() << ", now translation:" << render.CamTranslation();
}

void MainWindow::on_Render_clicked()
{
  QSize s = ui->graphicsView->size();
  s.setWidth(s.width()-30);
  s.setHeight(s.height()-30);
  render.Resize(s);
  qDebug() << "resized";

  Geometry geo1;
  geo1.name="Triangle";
  {
    VertexInfo v1; v1.p=QVector3D(-0.5, -1.5, -1.0);
    VertexInfo v2; v2.p=QVector3D(-0.5,  1.5, -1.0);
    VertexInfo v3; v3.p=QVector3D( 0.5,  0.0, -2.0);

    geo1.vecs.push_back(v2);
    geo1.vecs.push_back(v1);
    geo1.vecs.push_back(v3);

    geo1.SetNormal();
  }
  geo1.ambient = QVector4D(1.0, 0.0, 0.0, 0.8);
  geo1.specular = QVector4D(0.8, 1.0, 0.8, 0.8);
//  qDebug() << "before: " << geo1;
  render.AddGeometry(geo1);

  Geometry geo2;
  geo2.name="Rect";
  {
    VertexInfo v1; v1.p=QVector3D( 0.5, -0.5, -1.5);
    VertexInfo v2; v2.p=QVector3D(-0.5, -0.5, -1.5);
    VertexInfo v3; v3.p=QVector3D(-0.5,  0.5, -1.5);
    VertexInfo v4; v4.p=QVector3D( 0.5,  0.5, -1.5);

    geo2.vecs.push_back(v2);
    geo2.vecs.push_back(v1);
    geo2.vecs.push_back(v4);
    geo2.vecs.push_back(v3);

    geo2.SetNormal();
  }
  geo2.ambient = QVector4D(0.0, 1.0, 0.0, 0.8);
  geo2.specular = QVector4D(1.0, 0.8, 0.8, 0.8);
  render.AddGeometry(geo2);

  NewFrame();
}

void MainWindow::NewFrame()
{
//  render.CamRotate(QVector3D(1, 0, 0), -10);
  uchar* frame = render.Render();
  QSize size= render.Size();

  QImage image(frame, size.width(), size.height(), size.width()*3, QImage::Format_RGB888);

  QGraphicsPixmapItem* item = new QGraphicsPixmapItem( QPixmap::fromImage( image ) );
  QGraphicsScene* scene = new QGraphicsScene;
  scene->addItem(item);
  ui->graphicsView->setScene( scene );
//  qDebug() << "now rotation: " << render.WorldRotation().toEulerAngles() << "(1, 0, 0) into" << render.WorldRotation().toRotationMatrix()*QVector3D(1, 0, 0);
//  qDebug() << "now rotation matrix=" << render.WorldRotation().toRotationMatrix();
}

void MainWindow::on_graphicsView_ResizeEvent(QResizeEvent *)
{
  QSize s = ui->graphicsView->size();
  s.setWidth(s.width()-30);
  s.setHeight(s.height()-30);
  render.Resize(s);
  NewFrame();
}

void MainWindow::on_graphicsView_MousePressEvent(QMouseEvent *event)
{
  QPointF pp = ui->graphicsView->mapToScene(event->pos());
//  qDebug() << "clicked @" << event->pos() << "on scene @" << pp;
//  qDebug() << render.DepthPixelAt(pp.x(), pp.y());
}

void MainWindow::on_bLoadOBJ_clicked()
{
  QString filePath = QFileDialog::getOpenFileName(this, "Load OBJ File", "", "Wavefront (*.obj)");
  if(filePath=="")
  {
    return;
  }
//  qDebug() << "filePath=" << filePath;
  QVector<QVector3D> pointList;
  QVector<QVector3D> nlist;
  QFile file(filePath);
  if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
  {
    return;
  }
  QTextStream in(&file);
  while(!in.atEnd())
  {
    QString c;
    in >> c;
    c=c.toUpper();
    if(c=="#") in.readLine();
    else if(c=="V")
    {
      float x,y,z;
      in >> x >> y >> z;
      pointList.push_back(QVector3D(x, y, z));
      nlist.push_back(QVector3D(0, 0, 0));
    }
  }

  float xmin=1e20, xmax=-1e20;
  float ymin=1e20, ymax=-1e20;
  float zmin=1e20, zmax=-1e20;

  for(int i=0; i<pointList.size(); i++)
  {
    if(pointList[i].x()<xmin) xmin=pointList[i].x();
    if(pointList[i].x()>xmax) xmax=pointList[i].x();

    if(pointList[i].z()<zmin) zmin=pointList[i].z();
    if(pointList[i].z()>zmax) zmax=pointList[i].z();

    if(pointList[i].y()<ymin) ymin=pointList[i].y();
    if(pointList[i].y()>ymax) ymax=pointList[i].y();
  }

  float xc=(xmin+xmax)/2;
  float yc=(ymin+ymax)/2;
  float zc=(zmin+zmax)/2;

  float size=xmax-xmin;
  if(ymax-ymin>size) size=ymax-ymin;
  if(zmax-zmin>size) size=zmax-zmin;

  for(int i=0; i<pointList.size(); i++)
  {
    pointList[i].setX((pointList[i].x()-xc)/size);
    pointList[i].setY((pointList[i].y()-yc)/size);
    pointList[i].setZ((pointList[i].z()-zc)/size);
  }

  in.seek(0);

  while(!in.atEnd())
  {
    QString c;
    in >> c;
    c=c.toUpper();
    if(c=="#") in.readLine();
    else if(c=="F")
    {
      int v1, v2, v3;
      in >> v1 >> v2 >> v3;
      QVector3D p1=pointList[v1-1];
      QVector3D p2=pointList[v2-1];
      QVector3D p3=pointList[v3-1];
      QVector3D n=QVector3D::crossProduct(p3-p2, p1-p2).normalized();
      nlist[v1-1]+=n;
      nlist[v2-1]+=n;
      nlist[v3-1]+=n;
    }
  }

  in.seek(0);

  while(!in.atEnd())
  {
    QString c;
    in >> c;
    c=c.toUpper();
    if(c=="#") in.readLine();
    else if(c=="F")
    {
      int v1, v2, v3;
      in >> v1 >> v2 >> v3;
      Geometry g;
      VertexInfo vi1(pointList[v1-1], nlist[v1-1].normalized()); g.vecs.push_back(vi1);
      VertexInfo vi2(pointList[v2-1], nlist[v2-1].normalized()); g.vecs.push_back(vi2);
      VertexInfo vi3(pointList[v3-1], nlist[v3-1].normalized()); g.vecs.push_back(vi3);

      g.ambient = QVector4D(0.0, 1.0, 0.0, 0.7);
      g.inner = QVector4D(0.0, 0.6, 0.0, 0.7);
      g.specular = QVector4D(1.0, 0.8, 0.8, 0.7);
      render.AddGeometry(g);
    }
  }

  qDebug() << "pointList.size()=" << pointList.size();

  NewFrame();
}
