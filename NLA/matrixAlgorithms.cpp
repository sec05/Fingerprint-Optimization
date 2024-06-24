// This file contains algorithms to factorize matricies or solve linear systems
#include "matrix.h"
#include <string>
#include <math.h>
using namespace NLA;

/*
Matrix** Matrix::classicalGramSchmidt()
{
    int m = rows;
    int n = columns;
    Matrix** QR = new Matrix*[2];
    QR[0] = new Matrix(m, n);
    QR[1] = new Matrix(n, n);
    Matrix *R = QR[1];
    Matrix *Q = QR[0];
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            Q->data[i][j] = data[i][j];
        }
        NLA::Vector* q_j =  Q->getColumn(j);
        for (int k = 0; k < j; k++)
        {
            NLA::Vector* q_k = Q->getColumn(k);
            double dot = q_k->dot(q_j);
            R->data[k][j] = dot;
            for (int i = 0; i < m; i++)
            {
                Q->data[i][j] -= dot * Q->data[i][k];
            }
        }
         
        double norm = q_j->dot(q_j);
       
        norm = sqrt(norm);
        R->data[j][j] = norm;
        for (int i = 0; i < m; i++)
        {
            Q->data[i][j] /= norm;
        }
        
    }

    return QR;
}
/*
This function is an implementation of the modified Gram-Schmidt algorithm for QR decomposition.
It takes in a matrix A and returns a matrix Q and a matrix R such that A = QR.
It returns a pointer to an array of matrices where the first matrix is Q and the second matrix is R.

Matrix** Matrix::modifiedGramSchmidt() // returning Q^T fix this
{

    int m = rows;
    int n = columns;
    Matrix **QR = (Matrix**) malloc(sizeof(Matrix *) * 2);
    QR[1] = new Matrix(n,n);
    Matrix *R = QR[1];
    Matrix *Q = copyMatrix();
    for (int i = 0; i < n; i++)
    {
        // r_ii = ||a_i||
        double r_ii = sqrt(Q->getColumn(i)->dot(Q->getColumn(i)));
        R->data[i][i] = r_ii;
        // q_i = a_i / r_ii
        for (int j = 0; j < m; j++)
        {
            Q->data[j][i] = Q->data[j][i] / r_ii;
        }
        for (int j = i + 1; j < n; j++)
        {
            // r_ij = q_i^T * a_j
            double r_ij = Q->getColumn(i)->dot(getColumn(j));
            R->data[i][j] = r_ij;
            // a_j = a_j - r_ij * q_i
            for (int k = 0; k < m; k++)
            {
                Q->data[k][j] = Q->data[k][j] - r_ij * Q->data[k][i];
            }
        }
    }
    QR[0] = Q->copyMatrix();
    delete Q;
    return QR;
}

/*
This algorithm uses a series of householder reflectors to turn a symmetric matrix into a real
tridiagonal one or a non symmetric into an upper hessenberg


void Matrix::householderUpperHessenberg() {
    int m = rows;
    for (int k = 0; k < m-2; ++k) {
        // x = A(k+1:m, k)
        Vector* temp = getColumn(k);
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
        v_k->add(x);
        delete x;

        // v_k /= ||v_k||_2
        v_k->makeUnitVector();

        // A(k+1:m, k:m) = A(k+1:m, k:m) - 2 * v_k * v_k^T * A(k+1:m, k:m)
        Matrix* subA = new Matrix(m - k - 1, m - k);
        // Fill subA with entries from A
        for (int i = k + 1; i < m; ++i) {
            for (int j = k; j < m; ++j) {
                subA->data[i - k - 1][j - k] = data[i][j];
            }
        }
        Matrix* oProd = v_k->outerProduct(v_k); 
        oProd->scale(2);

        Matrix* prod = oProd->multiply(subA);

        for (int i = k + 1; i < m; ++i) {
            for (int j = k; j < m; ++j) {
                data[i][j] -= prod->data[i - k - 1][j - k];
            }
        }

        // A(1:m, k+1:m) = A(1:m, k+1:m) - 2 * A(1:m, k+1:m) * v_k * v_k^T
        subA = new Matrix(m, m - k - 1);
        for (int i = 0; i < m; ++i) {
            for (int j = k + 1; j < m; ++j) {
                subA->data[i][j - k - 1] = data[i][j];
            }
        }

        prod = subA->multiply(oProd);
      
        for (int i = 0; i < m; ++i) {
            for (int j = k + 1; j < m; ++j) {
                data[i][j] -= prod->data[i][j - k - 1];
            }
        }

        delete subA;
        delete prod;
        delete v_k;
    }
}*/
