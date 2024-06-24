#include "matrix.h"
#include <string>
#include <math.h>
using namespace NLA;

Matrix** classicalGramSchmidt(Matrix* A);
Matrix** modifiedGramSchmidt(Matrix* A);
Matrix* householderUpperHessenberg(Matrix* A) ;
Matrix** francis(Matrix* A, int iterations);
Matrix** francisWithShifts(Matrix* A, int iterations);