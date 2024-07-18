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
	if(strcmp(argv[1],"-in") != 0){
		printf("Error: input format it \"-in path/to/file\"\n");
		return 0;
	}
	OPT::Optimizer *optimizer = new OPT::Optimizer();
	optimizer->handleInput(argv[2]);
	optimizer->getKBestColumns(5);
	optimizer->outputVariables();
	return 0;
}
