#include "optimizer.h"
#include "generator.h"
#include <omp.h>
#include <armadillo>
#include <string>
#include <vector>

using namespace OPT;

Optimizer::Optimizer(char *f)
{
    inputFile = f;
    generator = new Generator(f);
}

Optimizer::~Optimizer()
{
    delete generator;
}


arma::dvec GMRES(arma::dmat* A, arma::dvec* b, int k, double tol)
{
   arma::dmat Q = arma::dmat(b->n_rows,k+1);
   arma::dmat H = arma::dmat(k+1,k);
   arma::dvec be1 = arma::dvec(b->n_rows);
   be1.zeros();
   be1.at(0) = arma::norm(*b,2);
   H.zeros();
   for(int n = 0; n < k; n++){
        // arnoldi
        arma::dvec v = (*A)*Q.col(n);
        for(int j = 0; j <= n; j++){
            H.at(j,n) = arma::as_scalar(Q.col(j)*v);
            v -=  H.at(j,n)*Q.col(j);
        }
        H.at(n+1,n) = arma::norm(v,2);
        if(H.at(n+1,n) < tol) break;
        Q.col(n+1) = v/H.at(n+1,n);
   }
    // solve least squares via QR
        arma::dmat G,R;
        arma::qr(G,R,H);
        arma::dvec y;
        if(b->n_rows == 1) y = arma::solve(R,G.t()*be1.at(0));
        return Q*y;
}


arma::uvec DEIM(arma::dmat *A, int k)
{

    arma::uvec selections(k);

    // v = V(:,1)
    arma::dvec v = A->col(0);

    // p_1 = argmax(|v|)
    selections[0] = arma::abs(v).index_max();
    for (int j = 1; j < k; j++)
    {
        // v = V(:,j)
        v = A->col(j);

        // c = V(p,1:j-1)^{-1} v(p)
        arma::dmat subMatrix = A->cols(0, j - 1);
        subMatrix = subMatrix.rows(selections.rows(0,j-1));
        arma::dvec cvec = v.rows(selections.rows(0,j-1));
        arma::dvec c = arma::solve(subMatrix,cvec);

        // r = v - V(p,1:j-1)c
        subMatrix = A->cols(0, j - 1);
        arma::dvec r = v - (subMatrix * c);

        // p_j = argmax(|r|)
        selections[j] = arma::abs(r).index_max();
    }

    return selections;
}

arma::uvec Optimizer::getKBestColumns(int k)
{
    printf("getting %d columns\n",k);
    // compute A^TA
    arma::dmat* product = new arma::dmat(fingerprints->n_cols,fingerprints->n_cols); 
    *product = (arma::trans(*fingerprints) * (*fingerprints));
    // compute right singular vectors
    arma::dvec singularValues;
    arma::dmat rightSingularVectors;
    arma::eig_sym(singularValues, rightSingularVectors, *product, "dc");
    arma::blas::
    delete product;
    return DEIM(&rightSingularVectors, k);
}

std::vector<std::string> Optimizer::returnKColumnVariables(arma::uvec cols)
{
    // open variable vector file
    std::string vector = inputFile;
    vector += ".optv";
    vector = "Optimizer Output/" + vector;
    std::ifstream f;
    f.open(vector);

    std::vector<std::string> variables;
    int max = cols.max();
    for (int i = 0; i <= max; i++)
    {
        std::string line;
        std::getline(f, line);
        arma::uvec inVector = arma::find(cols == i);
        if (inVector.n_elem != 0)
        {
            variables.push_back(line);
        }
    }

    return variables;
}
