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
	f.open("runtime.txt");
	for (int i = 1; i <= 14; i++)
	{
		int size = pow(2, i) / 2;
		auto start = std::chrono::high_resolution_clock::now();
		OPT::Optimizer *optimizer = new OPT::Optimizer("_Ti.nn");
		optimizer->fingerprints = optimizer->generator->generate_fingerprint_matrix(size, 0, 1, size, 0, 1);
		arma::uvec cols = optimizer->getKBestColumns(9);
		cols.print("Best columns of A");
		std::vector<std::string> variables = optimizer->returnKColumnVariables(cols);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		f << i << "," << elapsed.count() << std::endl;
		delete optimizer;
	}
	f.close();
	return 0;
}
