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

	OPT::Optimizer *optimizer = new OPT::Optimizer("NiTi.nn");
	optimizer->fingerprints = optimizer->generator->generate_fingerprint_matrix(500, 0, 10, 500, 0, 10);
	optimizer->getKBestColumns(50);
	optimizer->outputVariables("out.txt");
	return 0;
}
