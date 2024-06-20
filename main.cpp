/*
 * main.cpp
 *
 *  Top-level file for calibration code. Reads command line input, and creates new instance of Calibration class.
 *  Calibration class handles everything else.
 */

#include <stdio.h>
#include "optimizer.h"

//read command line input.
int main(int argc, char **argv)
{
	printf("Started running!\n");
	OPT::Optimizer* optimizer = new OPT::Optimizer("_Ti.nn");
	return 0;
}

