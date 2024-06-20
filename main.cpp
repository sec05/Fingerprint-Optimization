/*
 * main.cpp
 *
 *  Top-level file for calibration code. Reads command line input, and creates new instance of Calibration class.
 *  Calibration class handles everything else.
 */


#include "optimizer.h"

//read command line input.
int main(int argc, char **argv)
{
	printf("Started running!\n");
	Optimizer* optimizer = new Optimizer("_Ti.nn");
	optimizer->generate_input_file(1,2,3,4,5,6);
	return 0;
}

