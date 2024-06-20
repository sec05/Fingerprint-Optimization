/*
This class is the driver class for the optimization program 
*/
#include "NLA/matrix.h"

namespace OPT{
class Optimizer{
    public:
    Optimizer(char*);
    ~Optimizer();

    NLA::Matrix* fingerprints;
    char* inputFile;

};
}
