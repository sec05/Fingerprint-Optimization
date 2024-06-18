#include "matrix.h"
#include "vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
using namespace NLA;

Matrix::Matrix(int m, int n) {
    rows = m;
    columns = n;
    data = (double **) malloc(m*sizeof(double *));
    for(int i = 0; i < m; i++){
        data[i] = (double *) malloc(n*sizeof(double));
        for(int j = 0; j < n; j++){
            data[i][j] = 0;
        }
    }
}

Matrix::~Matrix() {
    for(int i = 0; i < rows; i++){
        delete [] data[i];
    }
    delete [] data;
}

void Matrix::scale(double n) {
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            data[i][j] *= n;
        }
    }
}

void Matrix::add(Matrix* m) {
    if(rows != m->rows || columns != m->columns){
        printf("Error: Cannot add matricies! Dimensions do not match!\n");
        return;
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            data[i][j] += m->data[i][j];
        }
    }
}

void Matrix::subtract(Matrix* m) {
    if(rows != m->rows || columns != m->columns){
        printf("Error: Cannot subtract matricies! Dimensions do not match!\n");
        return;
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            data[i][j] += m->data[i][j];
        }
    }
}

Matrix* Matrix::multiply(Matrix* m) {
     if (columns != m->rows)
    {
        printf("Error: Matrix dimensions do not match\n");
        return NULL;
    }
    Matrix* C = new Matrix(rows, m->columns);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < m->columns; j++)
        {
            double sum = 0;
            for (int k = 0; k < m->rows; k++)
            {
                sum += data[i][k] * m->data[k][j];
            }
            C->data[i][j] = sum;
        }
    }
    return C;
}

double Matrix::frobeniusNorm() {
    double norm = 0;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            norm += data[i][j] * data[i][j];
        }
    }
    return norm;
}

// use fixed in place method
// https://www.geeksforgeeks.org/inplace-m-x-n-size-matrix-transpose/
void Matrix::transpose() {
  int size = rows*columns - 1; 
    int t; // holds element to be replaced,  
           // eventually becomes next element to move 
    int next; // location of 't' to be moved 
    int cycleBegin; // holds start of cycle 
    int i; // iterator 
    bitset<HASH_SIZE> b; // hash to mark moved elements 
  
    b.reset(); 
    b[0] = b[size] = 1; 
    i = 1; // Note that A[0] and A[size-1] won't move 
    while (i < size) 
    { 
        cycleBegin = i; 
        t = data[i]; 
        do
        { 
            // Input matrix [r x c] 
            // Output matrix  
            // i_new = (i*r)%(N-1) 
            next = (i*rows)%size; 
            swap(data[next], t); 
            b[i] = 1; 
            i = next; 
        } 
        while (i != cycleBegin); 
  
        // Get Next Move (what about querying random location?) 
        for (i = 1; i < size && b[i]; i++);
    }   
}

bool Matrix::equals(Matrix* m) {
    if(rows != m->rows || columns != m->columns){
        printf("Error: Cannot evalulate equality of matricies! Dimensions do not match!\n");
        return false;
    }
    for(int i = 0; i < row; i++){
        for(int j = 0; j < columns; j++){
            if(data[i][j] != m->data[i][j]) return false;
        }
    }
    return true;
}

NLA::Vector* Matrix::getRow(int m) {
    if(rows >= m){
        printf("Error: Row requested is larger than matrix size!");
        return NULL;
    }
    double* row = malloc(sizeof(double)*columns);
    for(int i = 0; i < columns; i++){
        row[i] = data[m][i];
    }
    NLA::Vector* v = new NLA::Vector(row,columns);
    delete [] row;
    return v;
}

NLA::Vector* Matrix::getColumn(int n) {
    if(columns >= n){
        printf("Error: Column requested is larger than matrix size!");
        return NULL;
    }
    double* column = malloc(sizeof(double)*rows);
    for(int i = 0; i < rows; i++){
        column[i] = data[i][n];
    }
    NLA::Vector* v = new NLA::Vector(column,rows);
    delete [] column;
    return v;
}
