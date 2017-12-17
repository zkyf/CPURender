#ifndef QGRAPHICSVIEWWITHMOUSEEVENT_H
#define QGRAPHICSVIEWWITHMOUSEEVENT_H

#include <QGraphicsView>
#include <QWidget>

class QGraphicsViewWithMouseEvent : public QGraphicsView
{
  Q_OBJECT
public:
  QGraphicsViewWithMouseEvent(QWidget* parent=nullptr);

signals:
  void MouseEvent(QMouseEvent*);
  void MousePressEvent(QMouseEvent*);
  void MouseReleaseEvent(QMouseEvent*);
  void MouseMoveEvent(QMouseEvent*);
  void KeyPressEvent(QKeyEvent*);
  void KeyReleaseEvent(QKeyEvent*);
  void ResizeEvent(QResizeEvent*);

protected:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void keyPressEvent(QKeyEvent* event) override;
  void keyReleaseEvent(QKeyEvent* event) override;
  void resizeEvent(QResizeEvent *event) override;
};

#endif // QGRAPHICSVIEWWITHMOUSEEVENT_H
