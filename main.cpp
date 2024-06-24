/*
 * main.cpp
 *
 *  Top-level file for calibration code. Reads command line input, and creates new instance of Calibration class.
 *  Calibration class handles everything else.
 */

#include <stdio.h>
#include "optimizer.h"
#include "NLA/matrix.h"
//read command line input.
int main(int argc, char **argv)
{
	//OPT::Optimizer* optimizer = new OPT::Optimizer("_Ti.nn");
	int m = 26;
	NLA::Matrix* mat = new NLA::Matrix(m,m,"random");
	mat->outputToFile("./Matrix Output/randomSymmetric.matrix");
	mat->householderUpperHessenberg();
	mat->outputToFile("./Matrix Output/tridiag.matrix");
	return 0;
}

