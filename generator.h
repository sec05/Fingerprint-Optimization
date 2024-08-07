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
        void readSelectedVariables(std::vector<arma::uvec>&);
        void generateBestSelections();
        void generateOptimizedInputFile();
        int numRadialFingerprints;
        double radialFingerprintsLowerBound;
        double radialFingerprintsUpperBound;
        int numBondFingerprints;
        double bondFingerprintsLowerBound;
        double bondFingerprintsUpperBound;
        bool verbose;
        int selections;
        bool outputSelections;
        int selectionMethod;
        int outputRadialBlocks;
        int outputAlphaks;
        // for each atom type map of n, alphas and m, alpha_ks
        std::vector<std::map<int, std::vector<double>>> selectedRadial, selectedBond;
        std::vector<std::vector<double>> finalAlphas, finalAlphaKs;
        std::vector<std::string> atomTypes;
        std::vector<int> ms, alphas, totalRadial, totalBond;
        std::vector<std::pair<int,int>> osAndns;
            std::vector<std::string> radialCombinations;
                std::vector<std::string> bondCombinations;
                std::string inputFile;
        std::string outputFile;
    private:
        LAMMPS_NS::PairRANN *calibrator;
        
        void generate_opt_inputs();
        std::map<int, std::pair<std::string, std::string>> *readFile();
        void greedySelection();
        void largestSpanningSelection();
    };
}
#endif