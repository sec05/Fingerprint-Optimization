#include "optimizer.h"
#include "generator.h"
#include "NLA.h"
#include "utils.h"
// #include <omp.h>
#include <armadillo>
#include <string>
#include <vector>

using namespace OPT;

Optimizer::Optimizer()
{
    generator = new Generator();
    std::vector<arma::uvec> selections;
}

Optimizer::~Optimizer()
{
    delete generator;
}

void Optimizer::getKBestColumns()
{   
    int k = generator->selections;
    if(generator->verbose) printf("Getting %d best columns\n", k);
    for (int i = 0; i < fingerprints.size(); i++)
    {
        selections.push_back(deterministicCUR(fingerprints.at(i),k,generator->ms.at(i),generator->totalRadial.at(i)));
    }
}

void Optimizer::outputVariables()
{
    generator->readSelectedVariables(selections);
    generator->generateBestSelections();
    generator->generateOptimizedInputFile();
}

void Optimizer::handleInput(char* path){
    generator->parseParameters(path);
    fingerprints = generator->generate_fingerprint_matrix();
}