//    Copyright 2017 Rainer Gemulla
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
#include <util/evaluation.h>

#include <mf/matrix/coordinate.h>
#include <mf/ap/als.h>
#include <mf/lapack/lapack_wrapper.h>
#include <mf/logger.h>
#include <mf/loss/loss.h>
#include <mf/loss/l2.h>
#include <mf/loss/nzsl.h>
#include <mf/loss/nzl2.h>
#include <mf/matrix/op/scale.h>
#include <mf/matrix/op/sums.h>


//#include <gsl/gsl_linalg.h>

// GSL seems to be much faster and also runs well multi-threaded
// if the following line is commented out, LAPACK will be used (slower, doesn't work well with multiple threads)
//#define ALS_USE_GSL

namespace mf {

namespace detail {
  
#ifdef ALS_USE_GSL
	/** Solves system AA*x=b for x using the GSL library.
	 *
	 * @param[in,out] A input matrix (will be destroyed)
	 * @param[in,out] b right hand side (will be overwritten with result)
	 * @param[in,out] x temporary workspace; vector (length n)
	 * @param[in,out] V temporary workspace; matrix (n x n)
	 * @param[in,out] S temporary workspace; vector (length n)
	 */
	inline void llsGsl(gsl_matrix* A, gsl_vector* b, gsl_vector* x, gsl_matrix* V, gsl_vector* S) {
		gsl_linalg_SV_decomp(A, V, S, x);
		gsl_linalg_SV_solve(A, V, S, b, x);
		std::memmove(b->data, x->data, A->size2*sizeof(double));
	}
#endif

	void alsNzsl_w(const SparseMatrix& v, DenseMatrix& w, const DenseMatrixCM& h,
			const std::vector<mf_size_type>& nnz1, mf_size_type nnz1offset,
			double lambda, AlsRegularizer regularizer) {
		const SparseMatrix::index_array_type& vIndex1 = rowIndexData(v);
		const SparseMatrix::index_array_type& vIndex2 = columnIndexData(v);
		const SparseMatrix::value_array_type& vValues = v.value_data();

		mf_size_type r = w.size2();
#ifdef ALS_USE_GSL
		DenseMatrix A(r,r);
		gsl_vector* x = gsl_vector_alloc(r);
		gsl_matrix* V = gsl_matrix_alloc(r, r);
		gsl_vector* S = gsl_vector_alloc(r);

		gsl_matrix Agsl;
		Agsl.size1 = r;
		Agsl.size2 = r;
		Agsl.tda = r;
		Agsl.data = A.data().begin();
		Agsl.block = NULL;
		Agsl.owner = 0;

		gsl_vector bgsl;
		bgsl.size = r;
		bgsl.stride = 1;
		bgsl.block = NULL;
		bgsl.owner = 0;
#else
		DenseMatrixCM A(r,r);
		// working arrays
		std::pair<clapack::integer, clapack::integer> workSize = llsWork(r ,r);
		boost::numeric::ublas::vector<double> work(workSize.first);
		boost::numeric::ublas::vector<double> s(r);
		boost::numeric::ublas::vector<clapack::integer> iwork(workSize.second);
#endif


		// iterate over the rows
		mf_size_type p = 0;
		mf_size_type nnz = v.nnz();
		while (p < nnz) {
			// get the next row
			mf_size_type i = vIndex1[p];
			boost::numeric::ublas::matrix_row<DenseMatrix> row(w, i);

			// A will hold the coefficient matrix, row the rhs (and finally the result)
			double d = regularizer == ALS_L2 ? lambda : lambda * nnz1[i + nnz1offset];
			A.clear();
			for (mf_size_type k=0; k<r; k++) A(k,k) = d;
			std::fill(row.begin(), row.end(), 0.);

			// scan all entries in the current row and update A / rhs
			for( ; p < nnz && vIndex1[p] == i; p++) {
				mf_size_type j = vIndex2[p];
				boost::numeric::ublas::matrix_column<const DenseMatrixCM> col(h, j);
				A += outer_prod(col, col);
				row += col*vValues[p];
			}

			// find best fit (row will be overwritten)
#ifdef ALS_USE_GSL
			Agsl.data = A.data().begin();
			bgsl.data = &row[0];
			llsGsl(&Agsl, &bgsl, x, V, S);
#else
			lls(A, &row[0], s, work, iwork); // W is row-major, so this works
#endif
		}

#ifdef ALS_USE_GSL
		gsl_vector_free(x);
		gsl_matrix_free(V);
		gsl_vector_free(S);
#endif
	}

	void alsNzsl_h(const SparseMatrixCM& vc, const DenseMatrix& w, DenseMatrixCM& h,
			const std::vector<mf_size_type>& nnz2, mf_size_type nnz2offset,
			double lambda, AlsRegularizer regularizer) {
		const SparseMatrixCM::index_array_type& vIndex1 = rowIndexData(vc);
		const SparseMatrixCM::index_array_type& vIndex2 = columnIndexData(vc);
		const SparseMatrixCM::value_array_type& vValues = vc.value_data();

		mf_size_type r = w.size2();

#ifdef ALS_USE_GSL
		DenseMatrix A(r,r);
		gsl_vector* x = gsl_vector_alloc(r);
		gsl_matrix* V = gsl_matrix_alloc(r, r);
		gsl_vector* S = gsl_vector_alloc(r);

		gsl_matrix Agsl;
		Agsl.size1 = r;
		Agsl.size2 = r;
		Agsl.tda = r;
		Agsl.data = A.data().begin();
		Agsl.block = NULL;
		Agsl.owner = 0;

		gsl_vector bgsl;
		bgsl.size = r;
		bgsl.stride = 1;
		bgsl.block = NULL;
		bgsl.owner = 0;

#else
		DenseMatrixCM A(r,r);
		// working arrays
		std::pair<clapack::integer, clapack::integer> workSize = llsWork(r ,r);
		boost::numeric::ublas::vector<double> work(workSize.first);
		boost::numeric::ublas::vector<double> s(r);
		boost::numeric::ublas::vector<clapack::integer> iwork(workSize.second);
#endif

		// iterate over the columns
		mf_size_type p = 0;
		mf_size_type nnz = vc.nnz();
		while (p < nnz) {
			// get the next column
			mf_size_type j = vIndex2[p];
			boost::numeric::ublas::matrix_column<DenseMatrixCM> col(h, j);

			// A will hold the coefficient matrix, col the rhs (and finally the result)
			double d = regularizer == ALS_L2 ? lambda : lambda * nnz2[j + nnz2offset];
			A.clear();
			for (mf_size_type k=0; k<r; k++) A(k,k) = d;
			std::fill(col.begin(), col.end(), 0);

			// scan all entries in the current column and update A / rhs
			for( ; p < nnz && vIndex2[p] == j; p++) {
				mf_size_type i = vIndex1[p];
				boost::numeric::ublas::matrix_row<const DenseMatrix> row(w, i);
				A += outer_prod(row, row);
				col += row*vValues[p];
			}

			// find best fit (col will be overwritten)
#ifdef ALS_USE_GSL
			Agsl.data = A.data().begin();
			bgsl.data = &col[0];
			llsGsl(&Agsl, &bgsl, x, V, S);
#else
			lls(A, &col[0], s, work, iwork); // W is row-major, so this works
#endif
		}
#ifdef ALS_USE_GSL
		gsl_vector_free(x);
		gsl_matrix_free(V);
		gsl_vector_free(S);
#endif
	}

}

double rescaleSimple(FactorizationData<>& data, AlsRegularizer regularizer){
	// when WH is a good approximation but W is much larger than H, ALS is unable to balance
	// these two matrices. Here we decrease W and increase H so that the product is maintained
	// but the sum of l2(w) + l2(h) is minimized; this greatly improves the result on
	// the training data. Note that ALS cannot do such a step because it involves updating
	// both W and H at the same time
	double factor,regW,regH;
	LOG4CXX_INFO(mf::detail::logger, "Starting simple rescaling");

	if (regularizer==ALS_L2){
		regW=l2(data.w);
		regH=l2(data.h);
	}else{
		regW = nzl2(data.w, *data.nnz1, data.nnz1offset);
		regH = nzl2(data.h, *data.nnz2, data.nnz2offset);
	}
	factor = sqrt( sqrt(regH/regW) ); // the inner square root is the x that minimizes of x*l2w + 1/x*l2h
	LOG4CXX_INFO(mf::detail::logger, "Rescale factor f = " << factor);
	data.w *= factor;
	data.h /= factor;
	LOG4CXX_INFO(mf::detail::logger, "Finished simple rescaling");
	return factor;
}

boost::numeric::ublas::vector<double> rescaleOptimal(FactorizationData<>& data,AlsRegularizer regularizer){
	boost::numeric::ublas::vector<double> factor(data.r),regW,regH;
	LOG4CXX_INFO(mf::detail::logger, "Starting optimal rescaling");

	if (regularizer==ALS_L2){
		regW=squaredSums2(data.w);
		regH=squaredSums1(data.h);
	}else{
		regW=nzl2SquaredSums2(data.w,*data.nnz1, data.nnz1offset);
		regH=nzl2SquaredSums1(data.h,*data.nnz2, data.nnz2offset);
	}
	for (mf_size_type k=0; k<data.r; k++) {
		factor[k] = sqrt( sqrt(regH[k] / regW[k]) );
	}
	LOG4CXX_INFO(mf::detail::logger, "Rescale factors = " << factor);
	mult2(data.w, factor);
	div1(data.h, factor);
	LOG4CXX_INFO(mf::detail::logger, "Finished optimal rescaling");

	return factor;
}

void alsNzsl(FactorizationData<>& data, unsigned epochs, Trace& trace,
		double lambda, AlsRegularizer regularizer, BalanceType type, BalanceMethod method, FactorizationData<>* testData) {

	BOOST_ASSERT(data.vc != NULL);
	NzslLoss testLoss; // to be used only if testData are provided

	using namespace boost::numeric::ublas;
	LOG4CXX_INFO(detail::logger, "Starting ALS ("
			<< "with balance type : "
			<< (type == BALANCE_NONE ? "None" : "")
			<< (type == BALANCE_L2 ? "L2" : "")
			<< (type == BALANCE_NZL2 ? "Nzl2" : "")
			<< " and balance method : "
			<< (method == BALANCE_SIMPLE ? "Simple" : "")
			<< (method == BALANCE_OPTIMAL ? "Optimal" : "")
			<< "for nonzero squared loss and "
			<< (regularizer == ALS_L2 ? "L2" : "NZL2") << "(" << lambda << ")");

	// initialize
	double timeLoss=0;
	rg::Timer t;
	t.start();
	double currentLoss=nzsl(data.v, data.w, data.h);
	if (lambda > 0){
		if (regularizer == ALS_L2) {
			currentLoss += lambda*(l2(data.w) + l2(data.h));
		}else{
			currentLoss += lambda*(nzl2(data.w,*data.nnz1,data.nnz1offset)+nzl2(data.h,*data.nnz2,data.nnz2offset));
		}
	}
	t.stop();
	timeLoss=t.elapsedTime().nanos();
	trace.clear();

	AlsTraceEntry* entry;
	double currentTestLoss=0.0;
	double timeTestLoss=0.0;
	if (testData!=NULL){
		t.start();
		currentTestLoss=testLoss(*testData);
		t.stop();
		timeTestLoss=t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
		entry=new AlsTraceEntry(currentLoss, timeLoss,currentTestLoss,timeTestLoss);
	}
	else{
		entry=new AlsTraceEntry(currentLoss, timeLoss);

	}
	trace.add(entry);

	LOG4CXX_INFO(mf::detail::logger, "Loss: " << currentLoss << " (" << t << ")");

	// main loop
	for (mf_size_type epoch=0; epoch<epochs; epoch++) {

		boost::numeric::ublas::vector<double> factor;
		double timeBalancing = 0;

		// run epoch
		t.start();
		if (epoch % 2 == 0) {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating W)");
			detail::alsNzsl_w(data.v, data.w, data.h, *data.nnz1, data.nnz1offset, lambda, regularizer);
		} else {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating H)");
			detail::alsNzsl_h(*data.vc, data.w, data.h, *data.nnz2, data.nnz2offset, lambda, regularizer);
		}
		t.stop();
		double timeEpoch = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Finished epoch " << (epoch+1) << " (" << t << ")");

		// TODO: if balancing is used, loss computation below can be made more efficient
		// since we know the regularization loss already
		t.start();
		factor=balance(data, type, method);
		t.stop();
		timeBalancing=t.elapsedTime().nanos();

		// compute loss
		t.start();
		double currentLoss=nzsl(data.v, data.w, data.h);
		if (lambda > 0){
			if (regularizer == ALS_L2) {
				currentLoss += lambda*(l2(data.w) + l2(data.h));
			}else{
				currentLoss += lambda*(nzl2(data.w,*data.nnz1,data.nnz1offset)+nzl2(data.h,*data.nnz2,data.nnz2offset));
			}
		}
		t.stop();
		timeLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Loss: " << currentLoss << " (" << t << ")");

		AlsTraceEntry* entry;
		currentTestLoss=0.0;
		timeTestLoss=0.0;
		if (testData!=0){
			t.start();
			currentTestLoss=testLoss(*testData);
			t.stop();
			timeTestLoss=t.elapsedTime().nanos();
			LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
			// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
			entry=new AlsTraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, factor,timeBalancing, timeEpoch,currentTestLoss, timeTestLoss);
		}
		else{
			// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
			entry=new AlsTraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, factor,timeBalancing, timeEpoch);
		}
		trace.add(entry);
	}

	LOG4CXX_INFO(detail::logger, "Starting ALS ("
			<< "with balance type : "
			<< (type == BALANCE_NONE ? "None" : "")
			<< (type == BALANCE_L2 ? "L2" : "")
			<< (type == BALANCE_NZL2 ? "Nzl2" : "")
			<< " and balance method : "
			<< (method == BALANCE_SIMPLE ? "Simple" : "")
			<< (method == BALANCE_OPTIMAL ? "Optimal" : "")
			<< "for nonzero squared loss and "
			<< (regularizer == ALS_L2 ? "L2" : "NZL2") << "(" << lambda << ")");
}

}

