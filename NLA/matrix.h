#ifndef MATRIX_H
#define MATRIX_H
#include "vector.h"
#include <string>

namespace NLA{
    class Matrix{
        public:
        Matrix(int,int);
        Matrix(int,int,std::string);
        ~Matrix();

        int rows, columns;
        double** data;
        
        // matrix algorithms


        // arithemtic operations
        void scale(double);
        void add(Matrix*);
        void subtract(Matrix*);
        Matrix* multiply(Matrix*);
        double frobeniusNorm();

        // munipulation algorithms
        void transpose();
        bool equals(Matrix*);
        NLA::Vector* getRow(int);
        NLA::Vector* getColumn(int);
        

        //utility algorithms
        bool outputToFile(std::string);
        Matrix* copyMatrix();

        // math algorithms
        Matrix **classicalGramSchmidt();
        Matrix **modifiedGramSchmidt();
        void householderUpperHessenberg();
    };
}
#endif