#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);

  Geometry geo;
  geo.vecs.push_back(QVector3D(-0.5, 0, -1));
  geo.vecs.push_back(QVector3D(0, 0.5, -1));
  geo.vecs.push_back(QVector3D(0.5, 0, -1));

  geo.norms.push_back(QVector3D(-0.5, 0, -1));
  geo.norms.push_back(QVector3D(0, 0.5, -1));
  geo.norms.push_back(QVector3D(0.5, 0, -1));

  geo.texcoords.push_back(QVector2D(-0.5, 0));
  geo.texcoords.push_back(QVector2D(0, 0.5));
  geo.texcoords.push_back(QVector2D(0.5, 0));
  render.AddGeometry(geo);

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
