#include "optimizer.h"
#include "generator.h"
#include "NLA.h"
#include "util.h"
// #include <omp.h>
#include <armadillo>
#include <string>
#include <vector>

using namespace OPT;

Optimizer::Optimizer(char *f)
{
    inputFile = f;
    generator = new Generator(f);
    std::vector<arma::uvec> selections;
}

Optimizer::~Optimizer()
{
    delete generator;
}

void Optimizer::getKBestColumns(int k)
{
    printf("getting %d columns\n", k);
#pragma omp parallel for
    for (int i = 0; i < fingerprints.size(); i++)
    {
        // compute A^TA
        arma::dmat *product = new arma::dmat(fingerprints.at(i)->n_cols, fingerprints.at(i)->n_cols);
        *product = (*fingerprints.at(i)).t() * (*fingerprints.at(i));
        // compute right singular vectors
        arma::dvec singularValues;
        arma::dmat rightSingularVectors;
        arma::eig_sym(singularValues, rightSingularVectors, *product, "dc");
        delete product;
        selections.push_back(DEIM(&rightSingularVectors, k));
    }
}

void Optimizer::outputVariables(std::string path)
{
    // generate a list of atoms so we know what .optv file to open
    std::fstream reader;
    std::string file = inputFile;
    file+=".opt";
    file = "./Optimizer Output/"+file;
    reader.open(file, std::fstream::in);
    std::string atoms;
    std::getline(reader, atoms);
    std::getline(reader, atoms);
    reader.close();
    std::ofstream out;
    out.open(path);
    std::vector<std::string> atomTypes = LAMMPS_NS::utils::splitString(atoms);
    for (int i = 0; i < atomTypes.size(); i++)
    {
        std::string atom = atomTypes.at(i);
        arma::uvec cols = selections.at(i);
        // open variable vector file
        std::string vector = inputFile;
        vector += "."+atom+".optv";
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
        out << atom << std::endl;
        for(std::string variable : variables){
            out << variable << std::endl;
        }
    }
    out.close();
}
