#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QKeyEvent>
#include <QFileDialog>
#include "qgraphicsviewwithmouseevent.h"
#include "cpurenderer.h"
#include "mesh.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

private slots:
  void on_graphicsView_KeyReleaseEvent(QKeyEvent *);

  void on_Render_clicked();

  void NewFrame();

  void on_graphicsView_ResizeEvent(QResizeEvent *);

  void on_graphicsView_MousePressEvent(QMouseEvent *event);

  void on_bLoadOBJ_clicked();

  void on_bClear_clicked();

  void on_bReset_clicked();

  void on_bSmooth_clicked();

  void on_bSharp_clicked();

  void on_bOriginal_clicked();

private:
  Ui::MainWindow *ui;

  CPURenderer render;
  MyModel model;
  MyModel original;
};

#endif // MAINWINDOW_H
