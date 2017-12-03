#include "light.h"

void Light::SetDirectionalLight(QVector3D dir, QVector3D col)
{
	type = Directional;
	direction = dir;
	color = col;

	pos = QVector3D(0, 0, 0);
	angle2 = 0.0f;
	strength = 0.0f;
}

void Light::SetPointLight(QVector3D position, float str, QVector3D col)
{
	type = Point;
	pos = position;
	color = col;
	strength = str;

	direction = QVector3D(0, 0, 0);
	angle2 = 0;
}

void Light::SetSpotLisht(QVector3D position, QVector3D dir, GLfloat a, QVector3D col)
{

}

void Light::SetEnvLight(QVector3D col)
{
	type = Envoronment;
	color = col;
}
