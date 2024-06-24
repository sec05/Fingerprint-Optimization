#include "matrixAlgorithms.h"
using namespace NLA;


Matrix** classicalGramSchmidt(Matrix* A)
{
    int m = A->rows;
    int n = A->columns;
    Matrix** QR = new Matrix*[2];
    QR[0] = new Matrix(m, n);
    QR[1] = new Matrix(n, n);
    Matrix *R = QR[1];
    Matrix *Q = QR[0];
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            Q->data[i][j] = A->data[i][j];
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

//returnss a pointer to an array of matrices where the first matrix is Q and the second matrix is R.

Matrix** modifiedGramSchmidt(Matrix* A) // returning Q^T fix this
{

    int m = A->rows;
    int n = A->columns;
    Matrix **QR = (Matrix**) malloc(sizeof(Matrix *) * 2);
    QR[1] = new Matrix(n,n);
    Matrix *R = QR[1];
    Matrix *Q = A->copyMatrix();
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
            double r_ij = Q->getColumn(i)->dot(A->getColumn(j));
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
