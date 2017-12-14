#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);

  Geometry geo1;
  geo1.vecs.push_back(QVector3D(-0.5, -1.5, -1));
  geo1.vecs.push_back(QVector3D(-0.5, 1.5, -1));
  geo1.vecs.push_back(QVector3D(0.5, 0, -2));

  geo1.norms.push_back(QVector3D(-0.5, 0, -1));
  geo1.norms.push_back(QVector3D(0, 0.5, -1));
  geo1.norms.push_back(QVector3D(0.5, 0, -1));

  geo1.texcoords.push_back(QVector2D(-0.5, 0));
  geo1.texcoords.push_back(QVector2D(0, 0.5));
  geo1.texcoords.push_back(QVector2D(0.5, 0));

  geo1.ambient = QVector3D(1.0, 0.0, 0.0);
  render.AddGeometry(geo1);

  Geometry geo2;
  geo2.vecs.push_back(QVector3D(-0.5, 0, -1.5));
  geo2.vecs.push_back(QVector3D(0.5, 0.5, -1));
  geo2.vecs.push_back(QVector3D(0.5, -0.5, -1));

  geo2.norms.push_back(QVector3D(-0.5, 0, -1));
  geo2.norms.push_back(QVector3D(0, 0.5, -1));
  geo2.norms.push_back(QVector3D(0.5, 0, -1));

  geo2.texcoords.push_back(QVector2D(-0.5, 0));
  geo2.texcoords.push_back(QVector2D(0, 0.5));
  geo2.texcoords.push_back(QVector2D(0.5, 0));

  geo2.ambient = QVector3D(0.0, 1.0, 0.0);
  render.AddGeometry(geo2);

  uchar* frame = render.Render();
  QSize size= render.Size();

  QImage image(frame, size.width(), size.height(), QImage::Format_RGB888);

  QGraphicsPixmapItem* item = new QGraphicsPixmapItem( QPixmap::fromImage( image ) );
  QGraphicsScene* scene = new QGraphicsScene;
  scene->addItem(item);
  ui->graphicsView->setScene( scene );
}

MainWindow::~MainWindow()
{
  delete ui;
}
