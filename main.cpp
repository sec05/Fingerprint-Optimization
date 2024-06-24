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
	int m = 3;
	NLA::Matrix iden(m,m,"identity");
	double arr[] = {1,2,3};
	NLA::Vector v(arr,3);
	iden = iden * 2;
	(iden * v).outputToFile("product.txt");
	return 0;
}

