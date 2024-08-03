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
    int counter = 0;
    int k = generator->selections;
    if (generator->verbose)
        printf("Getting %d best columns\n", k);
    for (int i = 0; i < fingerprints.size(); i++)
    {
        for (int j = 0; j < generator->atomTypes.size(); j++)
        {
            // need to select radial bond subset for each combination
            int radialColSize = generator->totalRadial.at(i) / generator->atomTypes.size();
            int bondColSize = generator->totalBond.at(i) / generator->atomTypes.size();
            if(generator->verbose) printf("Fingerprint matrix for %s has %d cols\n",generator->atomTypes.at(j).c_str(),fingerprints.at(i)->n_cols);
            if(generator->verbose) printf("Grabbing column subset %d -> %d, and %d -> %d\n", j * radialColSize, (j + 1) * radialColSize-1 , (j + 1) * radialColSize + j*bondColSize ,( (j + 1) * radialColSize + (j + 1) * bondColSize)-1);
            
            arma::dmat A = arma::join_rows(fingerprints.at(i)->cols(j * radialColSize, (j + 1) * radialColSize-1 ), fingerprints.at(i)->cols((j + 1) * radialColSize + j*bondColSize , ((j + 1) * radialColSize + (j + 1) * bondColSize)-1));
            selections.push_back(deterministicCUR(A, k, generator->ms.at(counter), radialColSize));
            
            // now we update the idices to match original entries
            if (generator->atomTypes.size() > 1)
            {
                if(generator->verbose) printf("Updating the entries to match original matrix\n");
                for (k = 0; k < selections.at(counter).n_elem; k++)
                {
                    int elm = selections.at(counter).at(k);
                    if (elm < radialColSize)
                    {
                        selections.at(counter).at(k) += j * radialColSize;
                        continue;
                    }
                    if (elm >= radialColSize)
                    {
                        if (j == 0)
                            selections.at(counter).at(k) += radialColSize;
                        else
                            selections.at(counter).at(k) += j * radialColSize + j * bondColSize;
                    }
                }
            }
            counter++;
        }
    }
}

void Optimizer::outputVariables()
{
    if(generator->verbose) printf("Converting column indices to variables\n");
    generator->readSelectedVariables(selections);
    if(generator->verbose) printf("Finding the best selections from the chosen columns\n");
    generator->generateBestSelections();
    if(generator->verbose) printf("Outputting optimized input file %s\n",generator->outputFile.c_str());
    generator->generateOptimizedInputFile();
}

void Optimizer::handleInput(char *path)
{
    generator->parseParameters(path);
    fingerprints = generator->generate_fingerprint_matrix();
}