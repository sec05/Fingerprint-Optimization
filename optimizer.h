/*
This class is the driver class for the optimization program 
*/
#include <armadillo>

namespace OPT{
class Optimizer{
    public:
    Optimizer(char*);
    ~Optimizer();

    arma::dmat* fingerprints;
    char* inputFile;

};
}
