#include <vector.h>

namespace NLA{
    class Matrix{
        public:
        Matrix(int,int);
        ~Matrix();

        int rows, columns;
        double** data;

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

    };
}