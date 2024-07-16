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
	OPT::Optimizer *optimizer = new OPT::Optimizer(argv[3],atoi(argv[1]));
	optimizer->fingerprints = optimizer->generator->generate_fingerprint_matrix(100, 0, 10, 125, 0, 10, atoi(argv[1]));
	optimizer->getKBestColumns(50);
	optimizer->outputVariables(argv[2]);
	return 0;
}
