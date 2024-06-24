/*
 * main.cpp
 *
 *  Top-level file for calibration code. Reads command line input, and creates new instance of Calibration class.
 *  Calibration class handles everything else.
 */

#include <stdio.h>
#include "optimizer.h"
#include "NLA/matrix.h"
#include "NLA/matrixAlgorithms.h"
//read command line input.
int main(int argc, char **argv)
{
	//OPT::Optimizer* optimizer = new OPT::Optimizer("_Ti.nn");
	NLA::Matrix A = NLA::Matrix(3,3,"random symmetric");
	A.outputToFile("./Matrix Output/A.matrix");
	Matrix** EVs = francis(&A,100);
	EVs[0]->outputToFile("./Matrix Output/vals.matrix");
	EVs[1]->outputToFile("./Matrix Output/vecs.matrix");
	return 0;
}

