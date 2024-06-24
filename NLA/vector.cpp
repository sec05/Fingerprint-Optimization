#include "vector.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <iomanip> 

using namespace NLA;

Vector::Vector(double* data, int length) {
    dimension = length;
    components = (double *) malloc(length*sizeof(double));
    for(int i = 0; i < length; i++){
        components[i] = data[i];
    }
}

Vector::Vector(int length) {
    dimension = length;
    components = (double *) malloc(length*sizeof(double));
    for(int i = 0; i < length; i++){
        components[i] = 0;
    }
}

Vector::~Vector() {
    delete [] components;
}

void Vector::scale(double n){
    for(int i = 0; i < dimension; i++){
        components[i] *= n;
    }
}

NLA::Matrix& Vector::outerProduct(const Vector& v){
    Matrix* A = new Matrix(dimension,v.dimension);
    for(int i = 0; i < dimension; i++){
        for(int j = 0; j < v.dimension; j++){
            A->data[i][j] = components[i]*v.components[j];
        }
    }
    return *A;
}

Vector& Vector::operator*(const double d){
    scale(d);
    return *this;
}

Matrix& Vector::operator*(const Vector& v){
    return outerProduct(v);
}

void Vector::add(const Vector& v) {
    if(dimension != v.dimension){
        printf("Error: Unable to add vectors! Dimensions incompatible!\n");
        return;
    }
    for(int i = 0; i < dimension; i++){
        components[i]+=v.components[i];
    }
}

Vector& Vector::operator+(const Vector& v){
    add(v);
    return *this;
}

void Vector::subtract(const Vector& v) {
    if(dimension != v.dimension){
        printf("Error: Unable to subtract vectors! Dimensions incompatible!");
        return;
    }
    for(int i = 0; i < dimension; i++){
        components[i]-=v.components[i];
    }
}

Vector& Vector::operator-(const Vector& v){
    subtract(v);
    return *this;
}

double& Vector::operator()(const int m){
    if(m >= dimension || m < 0) printf("Error: Trying to access index larger than vector length!\n");
    return components[m];
}

double Vector::dot(Vector* v) {
    if(dimension != v->dimension){
        printf("Error: Unable to dot vectors! Dimensions incompatible!");
        return -INFINITY;
    }
    double sum = 0;
    for(int i = 0; i < dimension; i++){
        sum += components[i]*v->components[i];
    }
    return sum;
}

void Vector::makeUnitVector() {
    this->scale(1/sqrt(dot(this)));
}



void Vector::outputToFile(std::string file){
    std::ofstream f;
    f.open(file);
    f << std::setprecision(20);
    for(int i = 0; i < dimension; i++){
        f << components[i];
        std::string s = i == dimension - 1 ? "" : " ";
        f << s;
    }
    f<<std::endl;
}
