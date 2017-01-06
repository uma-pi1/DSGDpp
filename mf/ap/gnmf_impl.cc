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

#include <mf/ap/gnmf.h>
#include <mf/lapack/lapack_wrapper.h>
#include <mf/logger.h>

#include <mf/loss/nzsl.h>
#include <mf/loss/sl.h>
#include <mf/matrix/coordinate.h>
#include <mf/matrix/op/scale.h>
#include <mf/matrix/op/sums.h>
#include <mf/matrix/op/crossprod.h>

namespace mf {

namespace detail {
	inline void element_div_0(DenseMatrix& m, const DenseMatrix& by) {
		BOOST_ASSERT(m.size1() == by.size1());
		BOOST_ASSERT(m.size2() == by.size2());
		for (mf_size_type i = 0; i<m.size1(); i++) {
			for (mf_size_type j = 0; j<m.size2(); j++) {
				m(i,j) /= by(i,j);
				if ( std::isnan( m(i,j) ) || std::isinf( m(i,j) ) ) {
					m(i,j) = 0;
				}
			}
		}
	}

	inline void element_div_0(DenseMatrixCM& m, const DenseMatrixCM& by) {
		BOOST_ASSERT(m.size1() == by.size1());
		BOOST_ASSERT(m.size2() == by.size2());
		for (mf_size_type j = 0; j<m.size2(); j++) {
			for (mf_size_type i = 0; i<m.size1(); i++) {
				m(i,j) /= by(i,j);
				if ( std::isnan( m(i,j) ) || std::isinf( m(i,j) ) ) {
					m(i,j) = 0;
				}
			}
		}
	}

	inline void element_prod(DenseMatrix& m, const DenseMatrix& by) {
		BOOST_ASSERT(m.size1() == by.size1());
		BOOST_ASSERT(m.size2() == by.size2());
		for (mf_size_type i = 0; i<m.size1(); i++) {
			for (mf_size_type j = 0; j<m.size2(); j++) {
				m(i,j) *=  by(i,j);
			}
		}
	}

	inline void element_prod(DenseMatrixCM& m, const DenseMatrixCM& by) {
		BOOST_ASSERT(m.size1() == by.size1());
		BOOST_ASSERT(m.size2() == by.size2());
		for (mf_size_type j = 0; j<m.size2(); j++) {
			for (mf_size_type i = 0; i<m.size1(); i++) {
				m(i,j) *=  by(i,j);
			}
		}
	}

	// vc NEEDS to be sorted!
	void gnmf_h(const SparseMatrixCM& vc, const DenseMatrix& w, DenseMatrixCM& h) {
		const SparseMatrix::index_array_type& vIndex1 = rowIndexData(vc);
		const SparseMatrix::index_array_type& vIndex2 = columnIndexData(vc);
		const SparseMatrix::value_array_type& vValues = vc.value_data();
		mf_size_type r = w.size2();
		mf_size_type n = vc.size2();

		// compute W'V
		DenseMatrixCM wtv(r,n);
		wtv.clear();
		DenseMatrixCM::array_type& wtvValues = wtv.data();

		// temp helper vector
		boost::numeric::ublas::vector<double> col(r);

		// iterate over the columns, then rows
		mf_size_type p = 0;
		mf_size_type nnz = vc.nnz();
		while (p < nnz) {
			// get the next column
			mf_size_type j = vIndex2[p];
			col.clear();

			for( ; p < nnz && vIndex2[p] == j; p++) {
				mf_size_type i = vIndex1[p];
				boost::numeric::ublas::matrix_row<const DenseMatrix> row(w, i);
				col += row*vValues[p];
			}

			for (mf_size_type k=0; k<r; k++) wtvValues[j*r+k] = col[k];
		}

		// compute W'W
		DenseMatrix wtw = crossprod(w);

		// TODO: can we avoid this below and do it all in one step, directly updating H?
		// compute W'WH
		DenseMatrixCM wtwh(r,n);
		noalias(wtwh) = prod(wtw,h);

		// compute wtv <- W'V / W'WH
		element_div_0(wtv, wtwh);

		// update H <- H * W'V / W'WH
		element_prod(h, wtv);
	}

	// v NEEDS to be sorted!
	void gnmf_w(const SparseMatrix& v, DenseMatrix& w, const DenseMatrixCM& h) {
		const SparseMatrix::index_array_type& vIndex1 = rowIndexData(v);
		const SparseMatrix::index_array_type& vIndex2 = columnIndexData(v);
		const SparseMatrix::value_array_type& vValues = v.value_data();
		mf_size_type r = w.size2();
		mf_size_type m = v.size1();

		// compute VH'
		DenseMatrix vht(m,r);
		vht.clear();
		DenseMatrix::array_type& vhtValues = vht.data();

		// temporary helper vector
		boost::numeric::ublas::vector<double> row(r);

		// iterate over the columns
		mf_size_type p = 0;
		mf_size_type nnz = v.nnz();
		while (p < nnz) {
			// get the next row
			mf_size_type i = vIndex1[p];
			row.clear();

			for( ; p < nnz && vIndex1[p] == i; p++) {
				mf_size_type j = vIndex2[p];
				boost::numeric::ublas::matrix_column<const DenseMatrixCM> col(h, j);
				row += col*vValues[p];
			}

			for (unsigned k=0; k<r; k++) {
				vhtValues[i*r+k] = row[k];
			}
		}

		// compute HH'
		DenseMatrixCM hht = tcrossprod(h);

		// TODO: can we avoid this below and do it all in one step, directly updating W?
		// compute WHH'
		DenseMatrix whht(m,r);
		noalias(whht) = prod(w,hht);

		// compute vht <- VH' / WHH'
		element_div_0(vht, whht);

		// update W <- W * VH' / WHH'
		element_prod(w, vht);
	}

}

void gnmf(FactorizationData<>& data, unsigned epochs, Trace& trace, BalanceType type, BalanceMethod method, FactorizationData<>* testData) {
	using namespace boost::numeric::ublas;
	BOOST_ASSERT(data.vc != NULL);
	LOG4CXX_INFO(detail::logger, "Starting GNMF (SL)");

	// initialize
	double timeLoss = 0;
	rg::Timer t;
	t.start();
	NzslLoss testLoss;
	double currentLoss = sl(data.v, data.w, data.h);
	t.stop();
	timeLoss=t.elapsedTime().nanos();
	LOG4CXX_INFO(mf::detail::logger, "Loss: " << currentLoss << " (" << t << ")");

	trace.clear();
	TraceEntry* entry;
	double currentTestLoss = 0.0;
	double timeTestLoss = 0.0;
	if (testData != NULL){
		t.start();
		currentTestLoss = testLoss(*testData);
		t.stop();
		timeTestLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
		entry = new TraceEntry(currentLoss, timeLoss, currentTestLoss, timeTestLoss);
	} else {
		entry = new TraceEntry(currentLoss, timeLoss);
	}
	trace.add(entry);

	// main loop
	for (mf_size_type epoch=0; epoch<epochs; epoch++) {
		double timeBalancing = 0;

		// run epoch
		t.start();
		if (epoch % 2 == 0) {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating H)");
			detail::gnmf_h(*data.vc, data.w, data.h);
		} else {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating W)");
			detail::gnmf_w(data.v, data.w, data.h);
		}
		t.stop();
		double timeEpoch = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Finished epoch " << (epoch+1) << " (" << t << ")");

		t.start();
		balance(data, type, method);
		t.stop();
		timeBalancing=t.elapsedTime().nanos();

		// compute loss
		t.start();
		double currentLoss = sl(data.v, data.w, data.h);
		t.stop();
		timeLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Loss: " << currentLoss << " (" << t << ")");

		// update trace
		TraceEntry* entry;
		if (testData != NULL){
			t.start();
			currentTestLoss = testLoss(*testData);
			t.stop();
			timeTestLoss = t.elapsedTime().nanos();
			LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
			// currrently the time for balancing is not in the trace
			entry = new TraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, timeEpoch, currentTestLoss, timeTestLoss);
		}
		else {
			// currrently the time for balancing is not in the trace
			entry = new TraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, timeEpoch);
		}
		trace.add(entry);
	}

	LOG4CXX_INFO(detail::logger, "Finished GNMF (SL)");
}


}
