#include <cmath>
#include <mf/types.h>
#include <mf/lapack/lapack_wrapper.h>

namespace mf {

clapack::integer lu(DenseMatrixCM& A, boost::numeric::ublas::vector<clapack::integer>& ipiv) {
	clapack::integer m = A.size1();
	clapack::integer n = A.size2();
	ipiv.resize(std::min(m,n), false);
	clapack::integer lda = m;
	clapack::integer info = 0;
	clapack::dgetrf_(&m, &n, A.data().begin(), &lda, ipiv.data().begin(), &info);
	return info;
}

clapack::integer lpLu(DenseMatrixCM& A_LU, boost::numeric::ublas::vector<clapack::integer>& A_ipiv,
		double b[]) {
	BOOST_ASSERT(A_LU.size1() == A_LU.size2());
	clapack::integer n = A_LU.size1();
	clapack::integer nrhs = 1;
	clapack::integer lda = n;
	clapack::integer ldb = n;
	clapack::integer info = 0;
	clapack::dgetrs_((char*)"No transpose", &n, &nrhs, A_LU.data().begin(), &lda,
			A_ipiv.data().begin(), b, &ldb, &info);
	return info;
}

clapack::integer lp(DenseMatrixCM& A,
		boost::numeric::ublas::vector<clapack::integer>& ipiv,
		double b[]) {
	BOOST_ASSERT(A.size1() == A.size2());
	clapack::integer n = A.size1();
	clapack::integer lda = n;
	clapack::integer ldb = n;
	clapack::integer nrhs = 1;
	clapack::integer info;
	clapack::dgesv_(&n, &nrhs, A.data().begin(), &lda, ipiv.data().begin(), b, &ldb, &info);
	return info;
}

clapack::integer qrWork(clapack::integer m, clapack::integer n) {
	clapack::integer lda = m;
	double work; // output
	clapack::integer lwork = -1;
	clapack::integer info = 0;
	clapack::dgeqrf_(&m, &n, NULL, &lda, NULL, &work, &lwork, &info);
	return work;
}

clapack::integer qr(DenseMatrixCM& A, boost::numeric::ublas::vector<double>& tau) {
	clapack::integer m = A.size1();
	clapack::integer n = A.size2();
	tau.resize(std::min(m,n), false);
	clapack::integer lda = m;
	clapack::integer lwork = qrWork(m, n);
	double work[lwork];
	clapack::integer info = 0;
	clapack::dgeqrf_(&m, &n, A.data().begin(), &lda, tau.data().begin(), work, &lwork, &info);
	return info;
}

std::pair<clapack::integer,clapack::integer> llsWork(clapack::integer m, clapack::integer n,
		clapack::integer nrhs) {
	clapack::integer lda = m;
	clapack::integer ldb = std::max(n,m);
	double work = -1; // output
	clapack::integer lwork = -1;
	clapack::integer iwork = -1;// output
	clapack::integer info = 0;
	clapack::dgelsd_(&m, &n, &nrhs, NULL, &lda, NULL, &ldb, NULL, NULL, NULL,
			&work, &lwork, &iwork, &info);
    if (iwork < 0) { // lapack bug?
        mf_size_type minmn = std::min(m,n);
        clapack::integer ispec = 9;
        mf_size_type smlsiz = (mf_size_type)clapack::ilaenv_(&ispec, (char *)"DGELSD", (char *)"", NULL, NULL, NULL, NULL);
        mf_size_type nlvl = std::max(0, (int)(std::log( minmn/(smlsiz+1.) ) / std::log(2)) ) + 1;
        iwork = 3 * minmn * nlvl + 11 * minmn;
    }
	return std::pair<clapack::integer,clapack::integer>(work,iwork);
}

clapack::integer lls(DenseMatrixCM& A,
		double b[],
		boost::numeric::ublas::vector<double>& s,
		boost::numeric::ublas::vector<double>& work,
		boost::numeric::ublas::vector<clapack::integer>& iwork) {
	clapack::integer m = A.size1();
	clapack::integer n = A.size2();
	clapack::integer nrhs = 1;
	clapack::integer lda = m;
	clapack::integer ldb = std::max(n,m);
	double rcond = -1; // machine precision
	clapack::integer rank = 0;
	clapack::integer lwork = work.size();
	clapack::integer info = 0;
	s.resize(std::min(m,n), false);
	clapack::dgelsd_(&m, &n, &nrhs, A.data().begin(), &lda, b, &ldb,
			&s.data()[0], &rcond, &rank, work.data().begin(), &lwork, iwork.data().begin(),
			&info);
	return info;
}

}
