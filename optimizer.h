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

        std::vector<arma::dmat *> fingerprints;
        char *inputFile;
        Generator *generator;
        std::vector<arma::uvec> selections;

        void getKBestColumns(int);
        void outputVariables(std::string);
    };
}
