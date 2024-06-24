#include "matrix.h"
#include <string>
#ifndef VECTOR_H
#define VECTOR_H
namespace NLA {
class Matrix;
}
namespace NLA
{

    class Vector
    {
    public:
        Vector(double*, int);
        Vector(int);
        ~Vector();

        int dimension;
        double *components;

        void scale(double);
        void add(Vector*);
        void subtract(Vector*);
        double dot(Vector*);
        void makeUnitVector();
        NLA::Matrix* outerProduct(Vector*);
        void outputToFile(std::string);
    };
}
#endif