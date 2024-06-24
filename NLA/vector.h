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
        
        Vector& operator+(const Vector&);
        Vector& operator-(const Vector&);
        Vector& operator*(const double);
        Matrix& operator*(const Vector&);
        double& operator()(const int);
        
        double dot(Vector*);
        void makeUnitVector();
        void outputToFile(std::string);

    private:
        void scale(double);
        void add(const Vector&);
        void subtract(const Vector&);
        NLA::Matrix& outerProduct(const Vector&);
    };
}
#endif