#include "mainwindow.h"
#include "ui_mainwindow.h"

extern "C" void CudaGetRayTest(int xx, int yy, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right);
extern "C" void CudaIntersect(CudaRay ray);
extern "C" void CudaMonteCarloSampleTest(int xx, int yy, int w, int h, CudaVec camera, CudaVec up, CudaVec forward, CudaVec right);

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
  pulling = true;
  method = Original;

  ui->leNOL->setText(QString::number(render.nol));
  ui->leNOP->setText(QString::number(render.nop));
  ui->leSPSP->setText(QString::number(render.spsp));

  ui->rbOri->setChecked(true);
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
  }
  if(event->key()==Qt::Key_D)
  {
    render.CamRotate(render.CamUp(), -1);
  }
  if(event->key()==Qt::Key_W)
  {
    render.CamRotate(render.CamRight(), 1);
  }
  if(event->key()==Qt::Key_S)
  {
    render.CamRotate(render.CamRight(), -1);
  }

  const float tr = 0.05;
  if(event->key()==Qt::Key_J)
  {
    render.CamTranslate(-render.CamRight()*tr);
  }
  if(event->key()==Qt::Key_L)
  {
    render.CamTranslate(render.CamRight()*tr);
  }
  if(event->key()==Qt::Key_I)
  {
    render.CamTranslate(render.CamForward()*tr);
  }
  if(event->key()==Qt::Key_K)
  {
    render.CamTranslate(-render.CamForward()*tr);
  }
  if(event->key()==Qt::Key_U)
  {
    render.CamTranslate(render.CamUp()*tr);
  }
  if(event->key()==Qt::Key_O)
  {
    render.CamTranslate(-render.CamUp()*tr);
  }

  if(event->key()==Qt::Key_Space)
  {
    render.WorldRotate(QVector3D(0, 1, 0), 3);
  }

//  qDebug() << "now rotation: " << render.CamRotation().toEulerAngles() << ", now translation:" << render.CamTranslation();
  NewFrame();
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
  geo1.diffuse = QVector4D(1.0, 0.0, 0.0, 0.8);
  geo1.specular = QVector4D(0.8, 1.0, 0.8, 0.8);
//  qDebug() << "before: " << geo1;
  render.AddGeometry(geo1);

  Geometry geo2;
  geo2.name="Rect";
  {
    VertexInfo v1; v1.p=QVector3D( 0.5, -0.5, -1.5);
    VertexInfo v2; v2.p=QVector3D(-0.5, -0.5, -1.5);
    VertexInfo v3; v3.p=QVector3D(-0.5,  0.5, -1.5);
//    VertexInfo v4; v4.p=QVector3D( 0.5,  0.5, -1.5);

    geo2.vecs.push_back(v2);
    geo2.vecs.push_back(v1);
//    geo2.vecs.push_back(v4);
    geo2.vecs.push_back(v3);

//    QVector3D p1(0.1, 0.1, -1.5);
//    QVector3D p2(-1.0, 0.786, -1.5);
//    QVector3D p3(0, 0, -1);
//    qDebug() << geo2.IsInside(p1) << geo2.IsInside(p2) << geo2.IsInside(p3);

    geo2.SetNormal();
  }
  geo2.ambient = QVector4D(0.0, 1.0, 0.0, 0.8);
  geo2.diffuse = QVector4D(0.0, 1.0, 0.0, 0.8);
  geo2.specular = QVector4D(1.0, 0.8, 0.8, 0.8);
  geo2.emission = QVector4D(1.0, 0.8, 0.8, 0.8);
  render.AddGeometry(geo2);

  NewFrame();
}

void MainWindow::NewFrame()
{
//  for(int i=0; i<render.lightGeoList.size(); i++)
//  {
//    qDebug() << "lightgeo #: " << i << render.input[render.lightGeoList[i]].name << ":" << render.input[render.lightGeoList[i]].emission << render.input[render.lightGeoList[i]].emission.Strength();
//  }
  render.lights[0].pos=render.CamTranslation();
//  render.CamRotate(QVector3D(1, 0, 0), -10);
  uchar* frame = nullptr;
  switch(method)
  {
  case Original: frame = render.Render(); break;
  case MCRT: frame = render.MonteCarloRender(); break;
  }

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
  rendersize = s;
  s.setWidth(s.width()-30);
  s.setHeight(s.height()-30);
  render.Resize(s);
  NewFrame();
}

void MainWindow::on_graphicsView_MousePressEvent(QMouseEvent *event)
{
  QPointF pp = ui->graphicsView->mapToScene(event->pos());
  qDebug() << "clicked @" << event->pos() << "on scene @" << pp;
  if(method==MCRT)
  {
//    Ray ray = render.GetRay(pp.x(), pp.y());
//    ir.hp.valid=ir.valid;
//    CudaRay cray; cray.FromRay(ray);
//    CudaMonteCarloSampleTest(pp.x(), pp.y(), rendersize.width(), rendersize.height(),CudaVec(render.camera.translation()), CudaVec(render.camera.up()), CudaVec(render.camera.forward()), CudaVec(render.camera.right()));
//    CudaMonteCarloSampleTest(pp.x(), pp.y(), rendersize.width(), rendersize.height(), CudaVec(render.camera.translation()), CudaVec(render.camera.up()), CudaVec(render.camera.forward()), render.camera.right());
//    NewFrame();
//    if(ir.valid)
    {
//      qDebug() << ir.geo->name << ir.geo->reflectr;
//      render.MonteCarloSample(ir.hp, ray, 0, true);
    }
  }
  qDebug() << "color=" << pp << render.colorBuffer.buffer[(int)(pp.x()+pp.y()*render.Size().width())];
  qDebug() << "clicked";
  pulling = true;
  lastp = pp;
}

void MainWindow::on_bLoadOBJ_clicked()
{
  QString filePath = QFileDialog::getOpenFileName(this, "Load OBJ File", "", "Wavefront (*.obj)");
  if(filePath=="")
  {
    return;
  }
  render.ClearGeometry();
  model = LoadOBJ(filePath);
  model.Normalize();
  model.Triangulate();
//  model.Print();
  original = model;
//  model.Smooth();
  model.Render(&render);
  setWindowTitle(filePath);
  NewFrame();
}

void MainWindow::on_bClear_clicked()
{
  render.ClearGeometry();
  model.meshes.clear();
  original.meshes.clear();
  setWindowTitle("CPURenderer");
  NewFrame();
}

void MainWindow::on_bReset_clicked()
{
  render.ResetCam();
  render.CamTranslate(QVector3D(0, 0, 2));
  render.ResetWorld();
  NewFrame();
}

void MainWindow::on_bSmooth_clicked()
{
  render.ClearGeometry();
  model.Smooth();
  model.Render(&render);
  NewFrame();
}

void MainWindow::on_bSharp_clicked()
{
  render.ClearGeometry();
  model.Sharp();
  model.Render(&render);
  NewFrame();
}

void MainWindow::on_bOriginal_clicked()
{
  model = original;
  render.ClearGeometry();
  model.Render(&render);
  NewFrame();
}

void MainWindow::on_graphicsView_MouseReleaseEvent(QMouseEvent *event)
{
  pulling  = false;
}

void MainWindow::on_graphicsView_MouseMoveEvent(QMouseEvent *event)
{
  QPointF pp = ui->graphicsView->mapToScene(event->pos());
  QPointF df = pp-lastp;
//  qDebug() << pp;
  render.CamRotate(df.x()*QVector3D(0, 1, 0)+df.y()*QVector3D(1, 0, 0), QVector2D(df).length()*0.2);
  lastp=pp;
  NewFrame();
}

void MainWindow::on_rbOri_clicked()
{
  if(method!=Original)
  {
    method = Original;
    NewFrame();
  }
}

void MainWindow::on_rbMCRT_clicked()
{
  if(method!=MCRT)
  {
    method = MCRT;
    NewFrame();
  }
}

void MainWindow::on_MCRTRender_clicked()
{
  render.nol = ui->leNOL->text().toInt();
  render.nop = ui->leNOP->text().toInt();
  render.spsp = ui->leSPSP->text().toInt();

  NewFrame();
}
