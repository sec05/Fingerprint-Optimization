#include <armadillo>

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
    arma::uvec selections(k);

    // v = V(:,1)
    arma::dvec v = A->col(0);

    // p_1 = argmax(|v|)
    selections[0] = arma::abs(v).index_max();

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
    arma::qr(Q, R, A->head_cols(k));

    for (int j = k + 1; j < A->n_cols; j++)
    {
        arma::dvec rowNorms(R.n_rows + 1, arma::fill::zeros);
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

arma::uvec selectByImportanceScore(arma::dmat *A, int k)
{
    arma::uvec selections(k, arma::fill::zeros);
    for (int n = 0; n < k; n++)
    {
        arma::dvec scores(A->n_cols, arma::fill::zeros);

        // calculate right singular vectors of A
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
                scores.at(j) += rightSingularVectors.at(i, j) * rightSingularVectors.at(i, j);
            }
        }
        for (int i = 0; i <= n; i++)
            scores.at(selections.at(i)) = INT64_MIN;
        // for(int i = 0; i < n; i++) scores(selections.at(i)) = 0;
        //  Get best column
        int l = scores.index_max();
        selections.at(n) = l;
        arma::dvec A_l = A->col(l);

        // Perform Gram-Schmidt
        double colLNorm = arma::dot(A_l, A_l);
        if (colLNorm == 0)
            continue;
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

        // householder method
        /*for (int i = 0; i < A->n_cols; i++)
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
        }*/

        double sum = 0;
        for (int j = 0; j < A->n_cols; j++)
        {
            bool shouldSkip = false;
            for (int k = 0; k <= n; k++)
            {
                if (selections.at(k) == j)
                {
                    shouldSkip = true;
                    break;
                }
            }
            if (shouldSkip)
                continue;
            sum += arma::dot(A_l, A->col(j));
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