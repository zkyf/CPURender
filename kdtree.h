#ifndef KDTREE_H
#define KDTREE_H

#include <QPair>
#include "geometry.h"
#include "ray.h"

typedef QVector<Geometry> GeoList;

class KDTree
{
public:
  struct IR
  {
    GI geo;
    float d;
    bool valid;

    IR();

    bool operator<(const IR& b);
  };

  class Node
  {
  public:
    uchar axis; // 0=x, 1=y, 2=z
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
    float value;
    int depth;
    Node* left;
    Node* right;
    Node* parent;
    QVector<GI> gl;

    Node(Node* p = nullptr);
    ~Node();
    void Split();
    IR Hit(Ray ray);
  };

private:
  Node* root;

public:
  KDTree();
  KDTree(const GeoList& geos);
  ~KDTree();

  IR Intersect(Ray ray);
};

#endif // KDTREE_H
