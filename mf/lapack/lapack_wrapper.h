#ifndef MF_LAPACK_LAPACK_WRAPPER_H
#define MF_LAPACK_LAPACK_WRAPPER_H

#include <math.h>
#include <utility>

#include <mf/types.h>

namespace clapack {
#include <f2c.h>
#include <mf/lapack/blaswrap.h>
#include <mf/lapack/clapack.h>
#undef abs
#undef dabs
#undef min
#undef max
#undef dmin
#undef dmax
}

namespace mf {

/** Computes the LU factorization of A using the dgetrf method of LAPACK.
 *
 * @param[in,out] A coefficient matrix (in), LU factorization (out)
 * @param ipiv[out] pivot indexes used in the LU factorization (of size min(m,n))
 * @return LAPACK info value (0 = success)
 */
clapack::integer lu(DenseMatrixCM& A, boost::numeric::ublas::vector<clapack::integer>& ipiv);


/** Solves the unconstrained linear program Ax=b given the LU factorization (see mf::lu) of a
 * full-rank n-by-n matrix A using the dgetrs method of LAPACK.
 *
 * @param A_LU the LU factorization of A (as computed via mf::lu).
 * @param A_ipiv pivot indexes used in the LU factorization (as computed via mf::lu)
 * @param[in,out] b right-hand side (in), solution (out)
 * @return LAPACK info value (0 = success)
 */
clapack::integer lpLu(DenseMatrixCM& A_LU, boost::numeric::ublas::vector<clapack::integer>& A_ipiv,
		double b[]);


/** Solves the unconstrained linear program Ax=b with a full-rank n-by-n matrix A using the
 * dgesv method of LAPACK.
 *
 * @param[in,out] A coefficient matrix (in), LU factorization (out)
 * @param ipiv[out] pivot indexes used in the LU factorization (of size min(m,n))
 * @param[in,out] b right-hand side (in), solution (out)
 * @return LAPACK info value (0 = success)
 */
clapack::integer lp(DenseMatrixCM& A,
		boost::numeric::ublas::vector<clapack::integer>& ipiv,
		double b[]);


/** Determines the optimum size of the work array for the LAPACK method dgeqrf
 * (QR factorization).
 *
 * @param m number of rows
 * @param n number of columns
 * @return optimum size of the work array for the LAPACK method dgeqrf (number of double values)
 */
clapack::integer qrWork(clapack::integer m, clapack::integer n);


/** Computes the QR factorization of an m-by-n matrix A using the dgeqrf method of LAPACK.
 *
 * @param[in,out] A input matrix (in), QR decomposition (out)
 * @param[out] tau scalar factors of the elementary reflectors
 * @return
 */
clapack::integer qr(DenseMatrixCM& A, boost::numeric::ublas::vector<double>& tau);


/** Determines the optimum size of the work arrays for the LAPACK method dgelsd
 * (linear least squares).
 *
 * @param m number of rows of the coefficient matrix
 * @param n number of columns of the coefficient matrix
 * @param nrhs number of right-hand sides to solve for simultaneously
 * @return optimum size of the work array for the LAPACK method dgelsd (pair: number
 * of double for work, number of clapack::integer for iwork)
 */
std::pair<clapack::integer,clapack::integer> llsWork(clapack::integer m, clapack::integer n,
		clapack::integer nrhs = 1);


/** Solves the linear least squares problem "minimize 2-norm(Ax-b)" with a potentially
 * rank-deficient m-by-n matrix A using the dgelsd method of LAPACK.
 *
 * @param[in,out] A coefficient matrix (in), destroyed (out)
 * @param[in,out] b right-hand side (in), solution (out)
 * @param[out] s singular values of A in decreasing order
 * @param[in,out] work a work array of doubles (size should be determined by mf::llsWork)
 * @param[in,out] iwork a work array of integers (size should be determined by mf::llsWork)
 * @return LAPACK info value (0 = success)
 */
clapack::integer lls(DenseMatrixCM& A,
		double b[],
		boost::numeric::ublas::vector<double>& s,
		boost::numeric::ublas::vector<double>& work,
		boost::numeric::ublas::vector<clapack::integer>& iwork);

}

#endif
