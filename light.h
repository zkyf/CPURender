#ifndef LIGHT_H
#define LIGHT_H

#include <QVector3D>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>

class Light
{
	public:
		enum LightType
		{
			Directional = 0,
			Point = 1,
			Spot = 2,
			Envoronment = 3
		};

		Light() {}
		void SetDirectionalLight(QVector3D dir, QVector3D col = QVector3D(1.0, 1.0, 1.0));
		void SetPointLight(QVector3D position, float str  = 1.0, QVector3D col = QVector3D(1.0, 1.0, 1.0));
		void SetSpotLisht(QVector3D position, QVector3D dir, GLfloat a, QVector3D col = QVector3D(1.0, 1.0, 1.0));
		void SetEnvLight(QVector3D col = QVector3D(1.0, 1.0, 1.0));

		LightType type;
		QVector3D pos;
		QVector3D color;
		QVector3D direction;
		GLfloat angle2;
		GLfloat strength;
};

#endif // LIGHT_H
