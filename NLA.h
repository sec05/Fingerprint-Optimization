#include <armadillo>
#include <random>
#include <math.h>
#include "omp.h"

arma::dmat BSSSampling(arma::dmat &V, arma::dmat &R, int r, arma::uvec &selections, double ms, int offset)
{
    // setup for algorithm
    int n = V.n_rows;
    int k = V.n_cols;
    arma::dvec s(n, arma::fill::zeros);
    arma::dmat S(n, r, arma::fill::zeros);

    for (int i = offset; i < n; i++)
    {
        V.row(i) /= ms;
    }
    arma::dvec scores = arma::sum(arma::square(V), 1);

    double threshold = std::sqrt(static_cast<double>(k) / r);

    std::vector<std::pair<double, int>> scoreIndices;

    for (int i = 0; i < n; i++)
    {
        scoreIndices.push_back(std::make_pair(scores(i), i));
    }

    std::sort(scoreIndices.begin(), scoreIndices.end(), std::greater<std::pair<double, int>>());

    for (int i = 0; i < r; i++)
    {
        int index = scoreIndices[i].second;
        s(index) = 1.0 / (double)r;
        S(index, i) = std::sqrt(s(index));
        selections(i) = index;

        for (int j = i + 1; j < n - offset; j++)
        {
            scoreIndices[j].first = scoreIndices[i].first - scoreIndices[j].first;
        }
        std::sort(scoreIndices.begin() + i + 1, scoreIndices.end() - offset, std::greater<std::pair<double, int>>());
    }

    arma::dmat VTS = V.t() * S;
    arma::dmat RTS = R.t() * S;

    arma::dvec ew;
    arma::eig_sym(ew, VTS * VTS.t());
    if (ew.min() < 1 - threshold)
        printf("Spectral condition not met\n");
    if (arma::norm(RTS, "fro") > arma::norm(R, "fro"))
        printf("Frobenius condition not met\n");

    return S;
}

arma::dmat adaptiveCols(arma::dmat &A, arma::dmat &V, double alpha, int c, arma::uvec &selections, int offset, double ms, int o)
{
    // compute residual
    arma::dmat B = A - V * arma::pinv(V) * A;
    for (int i = o; i < B.n_cols; i++)
    {
        B.col(i) /= (ms * ms);
    }
    // calculate leverage scores for each row
    std::vector<double> ps;

    double froNorm = arma::norm(B, "fro");
    froNorm *= froNorm;
    // #pragma omp parallel for
    for (int i = 0; i < A.n_cols; i++)
    {
        double l = arma::norm(B.col(i));
        l *= l;
        ps.push_back(l / froNorm);
    }
    double sum = 0;

    for (double p : ps)
    {
        sum += p;
    }

    // assemble leverage score CDF
    std::vector<double> cdf(ps.size());
    std::partial_sum(ps.begin(), ps.end(), cdf.begin());

    // set up random generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    arma::dmat C(A.n_rows, c, arma::fill::zeros);

    // #pragma omp parallel for
    for (int i = 0; i < c; i++)
    {
        double randomValue = dis(gen);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), randomValue);
        int index = std::distance(cdf.begin(), it);
        C.col(i) = A.col(index);
        selections(i + offset) = index;
    }

    return C;
}

arma::uvec deterministicCUR(arma::dmat *A, int k, int ms, int offset)
{
    int n = 0;
    try
    {
        n = arma::rank(*A);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        printf("A is %d x %d\n", A->n_rows, A->n_cols);
    }

    if (k > n) 
        printf("Warning: k given is greater than rank(A)! rank(A) = %d \n", n);

    arma::uvec selections(k, arma::fill::value(-1));

    arma::dmat U, V, E;
    arma::dvec s;
    arma::svd_econ(U, s, V, *A);
    V = V.cols(0, k - 1);
    E = (*A) - (*A) * V * V.t();
    E = E.t();
    int a = k / 2, b = k / 2;
    if (k % 2 != 0)
        a++;
    arma::dmat S = BSSSampling(V, E, a, selections, ms, offset);
    arma::dmat C1 = (*A) * S;
    arma::dmat C2 = adaptiveCols(*A, C1, 1, b, selections, a, ms, offset);
    selections.save("selections.txt", arma::raw_ascii);
    return selections;
}
