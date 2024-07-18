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

void Optimizer::getKBestColumns(int k)
{
    int mode = 4;
    printf("Getting %d best columns\n", k);
    for (int i = 0; i < fingerprints.size(); i++)
    {
        switch (mode)
        {
        case 0:
            selections.push_back(DEIM(fingerprints.at(i),k));
            break;
        case 1:

            selections.push_back(QDEIM(fingerprints.at(i),k,0.9));
            break;
        case 2:
            selections.push_back(selectByImportanceScore(fingerprints.at(i),k,generator->ms.at(i),generator->totalRadial.at(i)));
            break;
        case 3:
            selections.push_back(farthestPointSampling(fingerprints.at(i),k));
            break;
        case 4:
            selections.push_back(deterministicCUR(fingerprints.at(i),k,generator->ms.at(i),generator->totalRadial.at(i)));
            break;
        case 5:
            selections.push_back(DAPDCX(fingerprints.at(i),k,0.5,5));
            break;
        default:
            break;
        }
    }
}

void Optimizer::outputVariables()
{
    generator->outputVariables(selections);
}

void Optimizer::handleInput(char* path){
    generator->parseParameters(path);
    fingerprints = generator->generate_fingerprint_matrix();
}