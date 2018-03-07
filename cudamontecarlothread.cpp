#include "cudamontecarlothread.h"

CudaMonteCarloThread::CudaMonteCarloThread(CudaVec* t, int w, int h, Camera3D cam)
{
  target = t;
  width = w;
  height = h;
  camera = cam;
}

void CudaMonteCarloThread::RenderFrame()
{
//  CudaRender(width, height, camera.translation(), camera.up(), camera.forward(), camera.right(), target);
  emit RenderEnded();
}

void CudaMonteCarloThread::run()
{
  QTimer::singleShot(30, this, &CudaMonteCarloThread::RenderFrame);
}
