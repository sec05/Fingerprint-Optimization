#include <armadillo>

arma::dvec GMRES(arma::dmat* A, arma::dvec* b, int k, double tol)
{
   arma::dmat Q = arma::dmat(b->n_rows,k+1);
   arma::dmat H = arma::dmat(k+1,k);
   arma::dvec be1 = arma::dvec(b->n_rows);
   be1.zeros();
   be1.at(0) = arma::norm(*b,2);
   H.zeros();
   for(int n = 0; n < k; n++){
        // arnoldi
        arma::dvec v = (*A)*Q.col(n);
        for(int j = 0; j <= n; j++){
            H.at(j,n) = arma::as_scalar(Q.col(j)*v);
            v -=  H.at(j,n)*Q.col(j);
        }
        H.at(n+1,n) = arma::norm(v,2);
        if(H.at(n+1,n) < tol) break;
        Q.col(n+1) = v/H.at(n+1,n);
   }
    // solve least squares via QR
        arma::dmat G,R;
        arma::qr(G,R,H);
        arma::dvec y;
        if(b->n_rows == 1) y = arma::solve(R,G.t()*be1.at(0));
        return Q*y;
}

arma::uvec DEIM(arma::dmat *A, int k)
{

    arma::uvec selections(k);

    // v = V(:,1)
    arma::dvec v = A->col(0);

    // p_1 = argmax(|v|)
    selections[0] = arma::abs(v).index_max();
    for (int j = 1; j < k; j++)
    {
        // v = V(:,j)
        v = A->col(j);

        // c = V(p,1:j-1)^{-1} v(p)
        arma::dmat subMatrix = A->cols(0, j - 1);
        subMatrix = subMatrix.rows(selections.rows(0,j-1));
        arma::dvec cvec = v.rows(selections.rows(0,j-1));
        arma::dvec c = arma::solve(subMatrix,cvec);

        // r = v - V(p,1:j-1)c
        subMatrix = A->cols(0, j - 1);
        arma::dvec r = v - (subMatrix * c);

        // p_j = argmax(|r|)
        selections[j] = arma::abs(r).index_max();
    }

    return selections;
}