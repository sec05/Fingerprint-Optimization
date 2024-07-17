#include <armadillo>
#include <random>
// #include "omp.h"

arma::dvec GMRES(arma::dmat *A, arma::dvec *b, int k, double tol)
{
    arma::dmat Q = arma::dmat(b->n_rows, k + 1);
    arma::dmat H = arma::dmat(k + 1, k);
    arma::dvec be1 = arma::dvec(b->n_rows);
    be1.zeros();
    be1.at(0) = arma::norm(*b, 2);
    H.zeros();
    for (int n = 0; n < k; n++)
    {
        // arnoldi
        arma::dvec v = (*A) * Q.col(n);
        for (int j = 0; j <= n; j++)
        {
            H.at(j, n) = arma::as_scalar(Q.col(j) * v);
            v -= H.at(j, n) * Q.col(j);
        }
        H.at(n + 1, n) = arma::norm(v, 2);
        if (H.at(n + 1, n) < tol)
            break;
        Q.col(n + 1) = v / H.at(n + 1, n);
    }
    // solve least squares via QR
    arma::dmat G, R;
    arma::qr(G, R, H);
    arma::dvec y;
    if (b->n_rows == 1)
        y = arma::solve(R, G.t() * be1.at(0));
    return Q * y;
}

arma::uvec DEIM(arma::dmat *A, int k)
{

    (*A) = arma::normalise(*A);
    arma::dvec s;
    arma::dmat V;
    arma::dmat p = (*A).t() * (*A);
    arma::eig_sym(s, V, p, "dc");
    arma::uvec selections(k, arma::fill::value(-1));

    // v = V(:,1)
    arma::dvec v = V.col(0);

    // p_1 = argmax(|v|)
    selections(0) = arma::abs(v).index_max();

    for (int j = 1; j < k; j++)
    {
        // v = V(:,j)
        v = V.col(j);

        // Extract the relevant submatrix and subvector
        arma::uvec subSelections = selections.subvec(0, j - 1);
        arma::dmat subMatrix = V.rows(subSelections);
        subMatrix = subMatrix.cols(0, j - 1);
        arma::dvec cvec = v.elem(subSelections);

        // c = V(p,1:j-1)^{-1} v(p)
        arma::dvec c = arma::inv(subMatrix) * cvec;

        // r = v - V(p,1:j-1)c
        arma::dvec r = v - V.cols(0, j - 1) * c;

        // p_j = argmax(|r|)
        selections[j] = arma::abs(r).index_max();
    }
    selections.save("selections.txt", arma::raw_ascii);
    return selections;
}

arma::uvec QDEIM(arma::dmat *A, int k, double tol)
{
    int kk = k;
    arma::dmat Q, R;
    R = R.t();
    arma::qr_econ(Q, R, A->head_cols(k));
    for (int j = k + 1; j < A->n_cols; j++)
    {
        arma::dvec rowNorms(R.n_rows + 1, arma::fill::zeros);

#pragma omp parallel for
        for (int i = 0; i < R.n_rows; i++)
        {
            rowNorms(i) = arma::norm(R.row(i), 2);
            rowNorms(i) *= rowNorms(i);
        }
        arma::dvec a = A->col(j);
        arma::dvec r = Q.t() * a;
        arma::dvec f = a - Q * r;
        arma::dvec c = Q.t() * f;
        f = f - Q * c;
        r = r + c;

        double rho = arma::norm(f, 2);
        arma::dvec q = f / rho;

        Q = arma::join_rows(Q, q);

        R = arma::join_rows(R, r);
        arma::dvec row(R.n_cols, arma::fill::zeros);

        row.at(R.n_cols - 1) = rho;
        R = arma::join_cols(R, row.t());

#pragma omp parallel for
        for (int i = 0; i < k; i++)
        {
            rowNorms(i) += r(i) * r(i);
        }
        rowNorms(k) = rho * rho;

        double FnormsR = arma::sum(rowNorms);
        double sigma = rowNorms.min();
        int i = rowNorms.index_min();

        if (sigma > (tol * tol) * (FnormsR - rowNorms(i)))
        {
            k++;
        } // no deflation
        else
        { // deflation
            if (i < k)
            {
                R.row(i) = R.row(k);
                Q.col(i) = Q.col(k);

                rowNorms(i) = rowNorms(k);
            }

            // delete minimum row norm of R
            Q = Q.head_cols(k);
            R = R.head_rows(k);
        }
    }

    return DEIM(&R, kk);
}

arma::uvec selectByImportanceScore(arma::dmat *A, int k, int ms, int offset, int mode = 1)
{
    arma::uvec selections(k, arma::fill::zeros);

    for (int n = 0; n < k; n++)
    {
        arma::dvec scores(A->n_cols, arma::fill::zeros);

        /* divide and conquer method*/
        arma::dvec singularValues;
        arma::dmat rightSingularVectors;
        arma::dmat product = (*A).t() * (*A);

        arma::eig_sym(singularValues, rightSingularVectors, product);

        // calcaulte importance scores of right singular vectors
        for (int j = 0; j < A->n_cols; j++)
        {
            scores.at(j) = 0;
            for (int i = 0; i < k - n; i++)
            {
                if (j >= offset)
                    scores.at(j) += (rightSingularVectors.at(j, i) * rightSingularVectors.at(j, i)) / ((ms - ((j - offset) % ms)) * (ms - ((j - offset) % ms)));
                else
                    scores.at(j) += (rightSingularVectors.at(j, i) * rightSingularVectors.at(j, i));
            }
        }

        for (int i = 0; i <= n; i++)
            scores.at(selections.at(i)) = INT64_MIN;
        scores.save("scores" + std::to_string(n) + ".txt", arma::raw_ascii);
        //  Get best column
        int l = scores.index_max();
        selections.at(n) = l;
        arma::dvec A_l = A->col(l);

        // Perform Gram-Schmidt
        if (mode == 1)
        {
            double colLNorm = arma::dot(A_l, A_l);
            if (colLNorm == 0)
                continue;
#pragma omp parallel for
            for (int j = 0; j < A->n_cols; j++)
            {
                if (arma::norm(A->col(j)) == 0)
                    continue;
                for (int i = 0; i <= n; i++)
                    if (selections.at(i) == j)
                        continue;
                arma::dvec A_l = A->col(l);
                A->col(j) -= A_l * (arma::dot(A_l, A->col(j)) / colLNorm);
            }
        }

        // householder method
        if (mode == 2)
        {
            for (int i = 0; i < A->n_cols; i++)
            {
                bool shouldSkip = false;
                for (int k = 0; k <= n; k++)
                {
                    if (selections.at(k) == i)
                    {
                        shouldSkip = true;
                        break;
                    }
                }
                if (shouldSkip)
                    continue;
                double alpha = arma::dot(A_l, A->col(i)) / arma::dot(A_l, A_l);
                arma::dvec u = A->col(i) - alpha * A->col(i);
                u /= arma::norm(u, 2);
                A->col(i) -= 2 * (u * arma::dot(u, A->col(i)));
            }
        }
    }
    selections.save("selections.txt", arma::raw_ascii);
    return selections;
}

arma::uvec farthestPointSampling(arma::dmat *A, int k)
{
    arma::uvec selections(k, arma::fill::zeros);
    for (int i = 0; i < k; i++)
    {
        int max = 0;
        for (int k = 0; k < i; k++)
        {
            for (int j = 0; j < A->n_cols; j++)
            {
                // check if in selections
                bool skip = false;
                for (int ii = 0; ii <= i; ii++)
                {
                    if (selections.at(ii) == j)
                        skip = true;
                }
                if (skip)
                    continue;
                double n = arma::norm(A->col(j) - A->col(selections.at(k)));
                if (n > max)
                {
                    max = n;
                    selections(i) = j;
                }
            }
        }
    }
    selections.save("selections.txt", arma::raw_ascii);
    return selections;
}

arma::dmat **randomizedSVD(arma::dmat *A, int k, int q)
{
    // over sampling parameter
    int l = 2 * k;

    // want to randomly sample all columns of A l times
    arma::dmat Y(A->n_rows, l, arma::fill::zeros);

    // setting up random generation
    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0, 1);

#pragma omp parallel for
    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < A->n_cols; j++)
        {
            Y.col(i) += dist(e2) * A->col(j);
        }
    }
    // using the power iteration method
    arma::dmat P = (*A) * A->t();
    for (int i = 0; i < q; i++)
        P *= P;

    Y = P * Y;

    // find Q
    arma::dmat Q, R;
    arma::qr(Q, R, Y);

    // form small matrix B
    arma::dmat B = Q.t() * (*A);

    // take SVD of B
    arma::dmat *U_hat = new arma::dmat();
    arma::dmat *V = new arma::dmat();
    arma::dvec S;
    arma::svd(*U_hat, S, *V, B);
    ;

    // form Q
    arma::dmat *U = new arma::dmat();
    (*U) = Q * (*U_hat);

    arma::dmat **r = (arma::dmat **)malloc(2 * sizeof(arma::dmat *));
    r[0] = U;
    r[1] = V;
    return r;
}

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
    printf("Running DCUR!\n");
    if (k > arma::rank(*A))
        printf("k given is greater than rank(A)! rank(A) = %d \n", arma::rank(*A));
    arma::uvec selections(k, arma::fill::value(-1));
    // deterministic SVD
    arma::dmat U, V, E;
    arma::dvec s;
    arma::svd(U, s, V, *A);
    V = V.cols(0, k - 1);
    E = (*A) - (*A) * V * V.t();
    E = E.t();

    arma::dmat S = BSSSampling(V, E, k / 2, selections, ms, offset);
    arma::dmat C1 = (*A) * S;
    arma::dmat C2 = adaptiveCols(*A, C1, 1, k / 2, selections, k / 2, ms, offset);
    selections.save("selections.txt", arma::raw_ascii);
    return selections;
}

arma::uvec DAPDCX(arma::dmat *A, int k, double delta, int l)
{
    arma::dmat E = *A;
    arma::uvec selections;
    selections.resize(0);
    while (selections.n_elem < k)
    {
        arma::dmat U, V;
        arma::dvec S;
        arma::svd_econ(U, S, V, E);

        int b = INT16_MAX;
        for (int i = 1; i <= k - selections.n_elem; i++)
        {
            if (S.at(i) >= delta * S.at(0))
                b = i;
        }
        int c = b;
        if (l < c)
            c = l;

        arma::uvec p = DEIM(&V, c);

        selections = arma::join_cols(selections, p);

        arma::dmat C = A->cols(p);
        E = *A - C * arma::pinv(C) * *A;
    }

    return selections;
}