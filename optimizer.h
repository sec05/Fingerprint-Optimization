/*
This class is the driver class for the optimization program
*/
#include <armadillo>
#include "generator.h"
namespace OPT
{
    class Optimizer
    {
    public:
        Optimizer(char *);
        ~Optimizer();

        arma::dmat *fingerprints;
        char *inputFile;
        Generator* generator;

        arma::uvec getKBestColumns(int);
        std::vector<std::string> returnKColumnVariables(arma::uvec);

    };
}
