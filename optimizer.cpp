#include "optimizer.h"
#include "generator.h"
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
    delete fingerprints;
}

arma::uvec DEIM(arma::dmat* A, int k){

    arma::uvec selections(k);

    // v = V(:,1)
    arma::dvec v = A->col(0);

    // p_1 = argmax(|v|)
    selections[0] = arma::abs(v).index_max();

    for(int j = 1; j < k; j++){
        // v = V(:,j)
        v = A->col(j);

        // c = V(p,1:j-1)^{-1} v(p)
        arma::dmat subMatrix = A->cols(0,j-1);
        subMatrix = subMatrix.rows(selections);
        arma::dvec cvec = v.rows(selections);
        arma::dvec c = arma::solve(subMatrix,cvec);

        //r = v - V(p,1:j-1)c
        subMatrix = A->cols(0,j-1);
        arma::dvec r = v - (subMatrix * c);

        //p_j = argmax(|r|)
        selections[j] = arma::abs(r).index_max();
    }

    return selections;
}   

arma::uvec Optimizer::getKBestColumns(int k){
    
    // compute A^TA
    arma::dmat* product = new arma::dmat(fingerprints->n_cols,fingerprints->n_cols);
    *product = (*fingerprints).t()*(*fingerprints);
    
    // compute right singular vectors
    arma::dvec singularValues;
    arma::dmat rightSingularVectors;
    arma::eig_sym(singularValues,rightSingularVectors,*product,"dc");

    return DEIM(&rightSingularVectors,k);
}

std::vector<std::string> Optimizer::returnKColumnVariables(arma::uvec cols){
    // open variable vector file
    std::string vector = inputFile;
    vector+=".optv";
    vector = "Optimizer Output/"+vector;
    std::ifstream f;
    f.open(vector);

    std::vector<std::string> variables;
    int max = cols.max();
    for(int i = 0; i <= max; i++){
        std::string line;
        std::getline(f,line);
        arma::uvec inVector = arma::find(cols == i);
        if(inVector.n_elem != 0){
            variables.push_back(line);
        }
    }

    return variables;
} 
