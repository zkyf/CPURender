#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QKeyEvent>
#include "qgraphicsviewwithmouseevent.h"
#include "cpurenderer.h"

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

private:
  Ui::MainWindow *ui;

  CPURenderer render;
};

#endif // MAINWINDOW_H
