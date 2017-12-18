#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);
  render.Resize(ui->graphicsView->size());
  Light light;
  light.SetPointLight(QVector3D(-1, -1, -1));
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
    render.CamRotate(QVector3D(0, 1, 0), 1);
    NewFrame();
  }
  if(event->key()==Qt::Key_D)
  {
    render.CamRotate(QVector3D(0, 1, 0), -1);
    NewFrame();
  }
  if(event->key()==Qt::Key_W)
  {
    render.CamRotate(QVector3D(1, 0, 0), 1);
    NewFrame();
  }
  if(event->key()==Qt::Key_S)
  {
    render.CamRotate(QVector3D(1, 0, 0), -1);
    NewFrame();
  }

  const float tr = 0.05;
  if(event->key()==Qt::Key_J)
  {
    render.CamTranslate(QVector3D(-1, 0, 0)*tr);
    NewFrame();
  }
  if(event->key()==Qt::Key_L)
  {
    render.CamTranslate(QVector3D(1, 0, 0)*tr);
    NewFrame();
  }
  if(event->key()==Qt::Key_I)
  {
    render.CamTranslate(QVector3D(0, 0, -1)*tr);
    NewFrame();
  }
  if(event->key()==Qt::Key_K)
  {
    render.CamTranslate(QVector3D(0, 0, 1)*tr);
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
    VertexInfo v1; v1.p=QVector3D(-0.5, -1.5, 0.05); geo1.vecs.push_back(v1);
    VertexInfo v2; v2.p=QVector3D(-0.5,  1.5, 0.05); geo1.vecs.push_back(v2);
    VertexInfo v3; v3.p=QVector3D( 0.5,  0.0, -2.0); geo1.vecs.push_back(v3);

    geo1.SetNormal();
  }
  geo1.ambient = QVector3D(1.0, 0.0, 0.0);
  qDebug() << "before: " << geo1;
  render.AddGeometry(geo1);

  Geometry geo2;
  geo2.name="Rect";
  {
    VertexInfo v1; v1.p=QVector3D( 0.5, -0.5, -1.5); geo2.vecs.push_back(v1);
    VertexInfo v2; v2.p=QVector3D(-0.5, -0.5, -1.5); geo2.vecs.push_back(v2);
    VertexInfo v3; v3.p=QVector3D(-0.5,  0.5, -1.5); geo2.vecs.push_back(v3);
    VertexInfo v4; v4.p=QVector3D( 0.5,  0.5, -1.5); geo2.vecs.push_back(v4);

    geo2.SetNormal();
  }
  geo2.ambient = QVector3D(0.0, 1.0, 0.0);
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
//  qDebug() << "now rotation: " << render.CamRotation().toEulerAngles() << ", now translation:" << render.CamTranslation();
}

void MainWindow::on_graphicsView_ResizeEvent(QResizeEvent *)
{
  QSize s = ui->graphicsView->size();
  s.setWidth(s.width()-30);
  s.setHeight(s.height()-30);
  render.Resize(s);
  NewFrame();
}
