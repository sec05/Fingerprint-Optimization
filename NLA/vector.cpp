#include "vector.h"
#include <stdlib.h>
#include <stdio.h>
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

void Vector::add(Vector* v) {
    if(dimension != v->dimension){
        printf("Error: Unable to add vectors! Dimensions incompatible!\n");
        return;
    }
    for(int i = 0; i < dimension; i++){
        components[i]+=v->components[i];
    }
}

void Vector::subtract(Vector* v) {
    if(dimension != v->dimension){
        printf("Error: Unable to subtract vectors! Dimensions incompatible!");
        return;
    }
    for(int i = 0; i < dimension; i++){
        components[i]-=v->components[i];
    }
}

double Vector::dot(Vector* v) {
    if(dimension != v->dimension){
        printf("Error: Unable to dot vectors! Dimensions incompatible!");
        return NULL;
    }
    double sum = 0;
    for(int i = 0; i < dimension; i++){
        sum += components[i]*v->components[i];
    }
    return sum;
}

void Vector::makeUnitVector() {
    scale(dot(this));
}
