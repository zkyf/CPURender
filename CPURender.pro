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
    kdtree.cpp \
    cugeometry.cpp

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
    kdtree.h \
    cugeometry.h

FORMS    += mainwindow.ui

# opencv
INCLUDEPATH += "C:/Libs/OpenCV31/opencv/build/include"
LIBS += "C:/Libs/OpenCV31/opencv/build/x64/vc14/lib/opencv_world310.lib"

#-------------------------------------------------

# CUDA settings
CUDA_SOURCES += montecarlosample.cu

CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0/"                # Path to cuda toolkit install
SYSTEM_NAME = x64                 # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64                    # '32' or '64', depending on your system
CUDA_ARCH = compute_61                 # Type of CUDA architecture
CUDA_CODE = sm_61
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += "$$CUDA_DIR/include" \
"C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc"

# library directories
QMAKE_LIBDIR += "$$CUDA_DIR/lib/x64"

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Add the necessary libraries
CUDA_LIB_NAMES += \
cublas \
cublas_device \
cuda \
cudadevrt \
cudart \
cudart_static \
cufft \
cufftw \
curand \
cusolver \
cusparse \
nppc \
nppi \
nppial \
nppicc \
nppicom \
nppidei \
nppif \
nppig \
nppim \
nppist \
nppisu \
nppitc \
npps \
nvblas \
nvcuvid \
nvgraph \
nvml \
nvrtc \
OpenCL \
kernel32 \
user32 \
gdi32 \
winspool \
comdlg32 \
advapi32 \
shell32 \
ole32 \
oleaut32 \
uuid \
odbc32 \
odbccp32 \
ucrt \
MSVCRT

for(lib, CUDA_LIB_NAMES) {
    CUDA_LIBS += $$lib.lib
}
for(lib, CUDA_LIB_NAMES) {
    NVCC_LIBS += -l$$lib
}
LIBS += $$NVCC_LIBS

# The following library conflicts with something in Cuda
QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrt.lib
QMAKE_LFLAGS_DEBUG   = /NODEFAULTLIB:msvcrtd.lib

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    DESTDIR = debug
    OBJECTS_DIR = debug/obj
    CUDA_OBJECTS_DIR = debug/cuda
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG -G -g $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -code=$$CUDA_CODE \
                      --compile -cudart static -g -DWIN32 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    DESTDIR = release
    OBJECTS_DIR = release/obj
    CUDA_OBJECTS_DIR = release/cuda
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -code=$$CUDA_CODE \
                    --compile -cudart static -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
