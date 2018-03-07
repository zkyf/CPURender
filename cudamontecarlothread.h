#ifndef CUDAMONTECARLOTHREAD_H
#define CUDAMONTECARLOTHREAD_H

#include <QObject>
#include <QThread>
#include <QTimer>
#include "camera3d.h"
#include "cugeometry.h"

class CudaMonteCarloThread : public QThread
{

  Q_OBJECT

public:
  CudaMonteCarloThread(CudaVec* t, int w, int h, Camera3D cam);

signals:
  void RenderEnded();

public slots:
  void RenderFrame();

private:
  CudaVec* target;
  int width;
  int height;
  Camera3D camera;

  void run();
};

#endif // CUDAMONTECARLOTHREAD_H
