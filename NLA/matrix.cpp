#include "matrix.h"
#include "vector.h"
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <fstream>
#include <random>
#include <iomanip> 

using namespace NLA;

Matrix::Matrix(int m, int n)
{
    rows = m;
    columns = n;
    data = (double **)malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++)
    {
        data[i] = (double *)malloc(n * sizeof(double));
        for (int j = 0; j < n; j++)
        {
            data[i][j] = 0;
        }
    }
}

Matrix::Matrix(int m, int n, std::string s)
{

    rows = m;
    columns = n;
    if (s == "identity")
    {
        data = (double **)malloc(m * sizeof(double *));
        for (int i = 0; i < m; i++)
        {
            data[i] = (double *)malloc(n * sizeof(double));
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                    data[i][j] = 1;
                else
                    data[i][j] = 0;
            }
        }
    }
    if (s == "random symmetric" && m == n)
    {
        data = (double **)malloc(m * sizeof(double *));
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_int_distribution<int> dist(0, 10);
        for(int i = 0; i < m; i++){
            data[i] = (double *)malloc(n * sizeof(double));
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = i; j < n; j++)
            {
                int t = dist(gen);
                data[i][j] = t;
                data[j][i] = t;
            }
        }

    }
    if(s == "random"){
        data = (double **)malloc(m * sizeof(double *));
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_int_distribution<int> dist(0, 10);
        for(int i = 0; i < m; i++){
            data[i] = (double *)malloc(n * sizeof(double));
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int t = dist(gen);
                data[i][j] = t;
            }
        }
    }
}

Matrix::~Matrix()
{
    for (int i = 0; i < rows; i++)
    {
        delete[] data[i];
    }
    delete[] data;
}

void Matrix::scale(double n)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            data[i][j] *= n;
        }
    }
}

void Matrix::add(const Matrix& m)
{
    if (rows != m.rows || columns != m.columns)
    {
        printf("Error: Cannot add matricies! Dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            data[i][j] += m.data[i][j];
        }
    }
}

Matrix& Matrix::operator+(const Matrix& A){
   add(A);
   return *this;
}

void Matrix::subtract(const Matrix& m)
{
    if (rows != m.rows || columns != m.columns)
    {
        printf("Error: Cannot subtract matricies! Dimensions do not match!\n");
        return;
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            data[i][j] += m.data[i][j];
        }
    }
}

Matrix& Matrix::operator-(const Matrix& A){
    subtract(A);
    return *this;
}

Matrix& Matrix::multiply(const Matrix& m)
{
    if (columns != m.rows)
    {
       printf("Error: Matrix dimensions do not match!\n");
    }
    Matrix *C = new Matrix(rows, m.columns);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < m.columns; j++)
        {
            double sum = 0;
            for (int k = 0; k < m.rows; k++)
            {
                sum += data[i][k] * m.data[k][j];
            }
            C->data[i][j] = sum;
        }
    }
    return *C;
}

Vector& Matrix::multiply(const Vector& v){
    if(columns != v.dimension)
    {
        printf("Error: Matrix columns and vector dimension do not match!\n");

    }
    Vector* p = new Vector(columns);

    for(int i = 0; i < rows; i++){
        double sum = 0;
        for(int j = 0; j < columns; j++){
            sum += data[i][j] * v.components[j];
        }
        p->components[i] = sum;
    }

    return *p;
}
Matrix& Matrix::operator*(const double d){
    scale(d);
    return *this;
}

Matrix& Matrix::operator*(const Matrix& m){
    return multiply(m);
}

Vector& Matrix::operator*(const Vector& v){
    return multiply(v);
}

double& Matrix::operator()(const int row, const int col){
    if(row >= rows || row < 0 || col >= columns || col < 0) printf("Error: Trying to access index larger than matrix!\n");
    return data[row][col];
}

double Matrix::frobeniusNorm()
{
    double norm = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            norm += data[i][j] * data[i][j];
        }
    }
    return norm;
}


// use fixed in place method
// https://www.geeksforgeeks.org/inplace-m-x-n-size-matrix-transpose/
void Matrix::transpose()
{
    int size = rows * columns - 1;
    double *t;          // holds element to be replaced,
                        // eventually becomes next element to move
    int next;           // location of 't' to be moved
    int cycleBegin;     // holds start of cycle
    int i;              // iterator
    std::bitset<128> b; // hash to mark moved elements

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
            next = (i * rows) % size;
            double *temp = t;
            t = data[next];
            data[next] = temp;
            delete temp;
            b[i] = 1;
            i = next;
        } while (i != cycleBegin);

        // Get Next Move (what about querying random location?)
        for (i = 1; i < size && b[i]; i++)
            ;
    }
}

bool Matrix::equals(Matrix *m)
{
    if (rows != m->rows || columns != m->columns)
    {
        printf("Error: Cannot evalulate equality of matricies! Dimensions do not match!\n");
        return false;
    }
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            if (data[i][j] != m->data[i][j])
                return false;
        }
    }
    return true;
}

NLA::Vector *Matrix::getRow(int m)
{
    if (rows <= m)
    {
        printf("Error: Row requested is larger than matrix size!");
        return NULL;
    }
    double *row = (double *)malloc(sizeof(double) * columns);
    for (int i = 0; i < columns; i++)
    {
        row[i] = data[m][i];
    }
    NLA::Vector *v = new NLA::Vector(row, columns);
    delete[] row;
    return v;
}

NLA::Vector *Matrix::getColumn(int n)
{
    if (columns <= n)
    {
        printf("Error: Column requested is larger than matrix size!");
        return NULL;
    }
    double *column = (double *)malloc(sizeof(double) * rows);
    for (int i = 0; i < rows; i++)
    {
        column[i] = data[i][n];
    }
    NLA::Vector *v = new NLA::Vector(column, rows);
    delete[] column;
    return v;
}

bool Matrix::outputToFile(std::string file)
{
   
    std::ofstream matrixFile;
    int r = 0;
    matrixFile.open(file);
    matrixFile << std::setprecision(20);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            matrixFile << data[i][j];
            std::string s = j == columns - 1 ? "" : " ";
            matrixFile << s;
        }
        matrixFile << std::endl;
    }
    matrixFile.close();
    return true;
}

Matrix *Matrix::copyMatrix()
{
    Matrix *A = new Matrix(rows, columns);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            A->data[i][j] = data[i][j];
        }
    }
    return A;
}
