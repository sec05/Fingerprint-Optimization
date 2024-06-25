#include "optimizer.h"
#include "generator.h"

using namespace OPT;

Optimizer::Optimizer(char *f)
{
    Generator* generator = new Generator(f);
    arma::dmat* fingerprints = generator->generate_fingerprint_matrix();
}

Optimizer::~Optimizer()
{
    delete[] fingerprints;
}

