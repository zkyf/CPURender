#-------------------------------------------------
#
# Project created by QtCreator 2017-12-04T01:54:36
#
#-------------------------------------------------

QT       += core gui xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = CPURender
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += _USE_MATH_DEFINES

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += main.cpp\
        mainwindow.cpp \
    cpurenderer.cpp \
    camera3d.cpp \
    light.cpp \
    transform3d.cpp \
    qgraphicsviewwithmouseevent.cpp \
    mesh.cpp \
    ray.cpp \
    geometry.cpp \
    kdtree.cpp

HEADERS  += mainwindow.h \
    cpurenderer.h \
    camera3d.h \
    light.h \
    transform3d.h \
    simplemesh.h \
    qgraphicsviewwithmouseevent.h \
    mesh.h \
    ray.h \
    geometry.h \
    kdtree.h

FORMS    += mainwindow.ui

# opencv
INCLUDEPATH += "C:/Libs/OpenCV31/opencv/build/include"
LIBS += "C:/Libs/OpenCV31/opencv/build/x64/vc14/lib/opencv_world310.lib"
