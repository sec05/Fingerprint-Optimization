/*
This class generates the new input file and the fingerprint matrix
*/
#include "pair_spin_rann.h"
#include <armadillo>
#include <map>

#ifndef GENERATOR_H
#define GENERATOR_H
#define DEBUG
namespace OPT
{
    class Generator
    {
    public:
        Generator();
        ~Generator();

        std::vector<arma::dmat *> generate_fingerprint_matrix();
        void parseParameters(char*);
        void outputVariables(std::vector<arma::uvec>&);
        int numRadialFingerprints;
        double radialFingerprintsLowerBound;
        double radialFingerprintsUpperBound;
        int numBondFingerprints;
        double bondFingerprintsLowerBound;
        double bondFingerprintsUpperBound;
        bool verbose;

        std::vector<int> ms, totalRadial, totalBond;
    private:
        LAMMPS_NS::PairRANN *calibrator;
        std::string inputFile;
        std::string outputFile;
        void generate_opt_inputs();
        std::map<int, std::pair<std::string, std::string>> *readFile();
    };
}
#endif