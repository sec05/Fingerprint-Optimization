/*
 * main.cpp
 *
 *  Top-level file for calibration code. Reads command line input, and creates new instance of Calibration class.
 *  Calibration class handles everything else.
 */

#include <stdio.h>
#include "optimizer.h"
#include <chrono>
#include <fstream>
#include <math.h>
// read command line input.
int main(int argc, char **argv)
{

	std::ofstream f;
	f.open("selections.txt");
	auto start = std::chrono::high_resolution_clock::now();
	OPT::Optimizer *optimizer = new OPT::Optimizer("_Ti.nn");
	optimizer->fingerprints = optimizer->generator->generate_fingerprint_matrix(500, 0, 1, 500, 0, 1);
	arma::uvec cols = optimizer->getKBestColumns(50);
	std::vector<std::string> variables = optimizer->returnKColumnVariables(cols);
	for(std::string variable : variables) f << variable << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	f << elapsed.count() << std::endl;
	f.close();
	return 0;
}
