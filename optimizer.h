#include <armadillo>
#include "generator.h"
namespace OPT
{
    class Optimizer
    {
    public:
        Optimizer();
        ~Optimizer();

        std::vector<arma::dmat *> fingerprints;
        Generator *generator;
        std::vector<arma::uvec> selections;
        void getKBestColumns();
        void outputVariables();
        void handleInput(char*);
        
    };
}
