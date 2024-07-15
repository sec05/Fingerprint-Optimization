#include "optimizer.h"
#include "generator.h"
#include "NLA.h"
#include "utils.h"
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
    printf("Getting %d best columns\n", k);
    for (int i = 0; i < fingerprints.size(); i++)
    {
        selections.push_back(selectByImportanceScore(fingerprints.at(i),k,generator->ms.at(i),generator->totalRadial.at(i),2));
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
        for (int i = 0; i < cols.n_elem; i++)
        {
            std::string line;
            int entry = 0;
            while(std::getline(f, line)){
                if(cols.at(i) == entry) 
                {
                    variables.push_back(line);
                    break;
                }
                else{
                    entry++;
                }
            }
            f.seekg(0);
        }
        out << atom << std::endl;
        for(std::string variable : variables){
            out << variable << std::endl;
        }
    }
    out.close();
}
