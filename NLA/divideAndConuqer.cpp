#include "matrix.h"
#include <string>
#include <math.h>

using namespace NLA;

/*
This function will use the divide and conquer method to find the eigenvectors of a tridiagonal matrix
Returns a matrix of eigenvectors.
*/

Matrix* divideAndConuqer(Matrix* A){
    // extraneous case
    if(A->rows == 1 && A->columns == 1) return new Matrix(1,1,"identity");



}

/*
This function will do the recursive divisions of T and compute Q and Lambda
*/

Matrix** divide(Matrix* A){

}