#include "matrixAlgorithms.h"
using namespace NLA;

/*
This algorithm uses a series of householder reflectors to turn a symmetric matrix into a real
tridiagonal one or a non symmetric into an upper hessenberg
*/


Matrix* householderUpperHessenberg(Matrix* A) {
    Matrix* T = A->copyMatrix();
    int m = T->rows;
    for (int k = 0; k < m-2; ++k) {
        // x = A(k+1:m, k)
        Vector* temp = T->getColumn(k);
        Vector* x = new Vector(m - k - 1);
        for (int i = k + 1; i < m; ++i) {
            x->components[i - k - 1] = temp->components[i];
        }
        delete temp;

        // v_k = sign(x(1))||x||_2e_1 + x
        double xNorm = sqrt(x->dot(x));
        int sign = (x->components[0] >= 0) ? 1 : -1;
        Vector* v_k = new Vector(x->dimension);
        v_k->components[0] = xNorm * sign;
        *v_k = *v_k + *x;
        delete x;

        // v_k /= ||v_k||_2
        v_k->makeUnitVector();

        // A(k+1:m, k:m) = A(k+1:m, k:m) - 2 * v_k * v_k^T * A(k+1:m, k:m)
        Matrix* subT = new Matrix(m - k - 1, m - k);
        // Fill subA with entries from A
        for (int i = k + 1; i < m; ++i) {
            for (int j = k; j < m; ++j) {
                subT->data[i - k - 1][j - k] = T->data[i][j];
            }
        }
        Matrix* oProd = &(*v_k * *v_k); 
        *oProd = *oProd * 2;

        Matrix* prod = &(*oProd * *subT);

        for (int i = k + 1; i < m; ++i) {
            for (int j = k; j < m; ++j) {
                T->data[i][j] -= prod->data[i - k - 1][j - k];
            }
        }

        // A(1:m, k+1:m) = A(1:m, k+1:m) - 2 * A(1:m, k+1:m) * v_k * v_k^T
        subT = new Matrix(m, m - k - 1);
        for (int i = 0; i < m; ++i) {
            for (int j = k + 1; j < m; ++j) {
                subT->data[i][j - k - 1] = T->data[i][j];
            }
        }

        prod = &(*subT * *oProd);
      
        for (int i = 0; i < m; ++i) {
            for (int j = k + 1; j < m; ++j) {
                T->data[i][j] -= prod->data[i][j - k - 1];
            }
        }

        delete subT;
        delete prod;
        delete v_k;
    }
    return T;
}