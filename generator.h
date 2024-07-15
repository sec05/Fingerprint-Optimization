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
        Generator(char *);
        ~Generator();

        std::vector<arma::dmat *> generate_fingerprint_matrix(int, double, double, int, double, double);
        int numRadialFingerprints;
        double radialFingerprintsLowerBound;
        double radialFingerprintsUpperBound;
        int numBondFingerprints;
        double bondFingerprintsLowerBound;
        double bondFingerprintsUpperBound;
        std::vector<int> ms, totalRadial, totalBond;
    private:
        LAMMPS_NS::PairRANN *calibrator;
        std::string inputFile;

        void generate_opt_inputs();
        void parse_parameters();
        std::map<int, std::pair<std::string, std::string>> *readFile();
    };
}
#endif