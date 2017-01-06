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
#include <mf/ap/lee01-gkl.h>
#include <mf/logger.h>
#include <mf/loss/loss.h>
#include <mf/matrix/op/sums.h>

#include <mf/loss/nzsl.h>
#include <mf/loss/nzl2.h>
#include <mf/loss/gkl.h>

namespace mf {

namespace detail {
	/** Updates h */
	void lee01Gkl_h(const SparseMatrix& v, const DenseMatrix& w, DenseMatrixCM& h,
			DenseMatrixCM& num, boost::numeric::ublas::vector<double>& denom) {
		//mf_size_type m = v.size1();
		mf_size_type n = v.size2();
		mf_size_type r = w.size2();

		// compute numerator: t(w) [ v / w*h ]
		num.resize(r, n, false);
		std::fill(num.data().begin(), num.data().end(), 0.);
		const SparseMatrix::index_array_type& index1 = rowIndexData(v);
		const SparseMatrix::index_array_type& index2 = columnIndexData(v);
		const SparseMatrix::value_array_type& values = v.value_data();
		for (mf_size_type p=0; p<v.nnz(); p++) {
			mf_size_type i = index1[p];
			mf_size_type j = index2[p];
			double x = values[p];

            // compute the inner product
            double wh = 0;
            for (mf_size_type z=0; z<r; z++) {
                wh += w(i,z) * h(z,j);
            }

            // update numerator
            double f = x / wh;
            for (mf_size_type z=0; z<r; z++) {
                num(z,j) += w(i,z) * f;
            }
		}

		// compute denominator
		denom = sums2(w);

		// update h
	    for (mf_size_type z=0; z<r; z++) {
	        for (mf_size_type j=0; j<n; j++) {
	            h(z,j) *= num(z,j) / denom[z];
	        }
	    }
	}

	/** Updates h */
	void lee01Gkl_h(const SparseMatrixCM& v, const DenseMatrix& w, DenseMatrixCM& h,
			DenseMatrixCM& num, boost::numeric::ublas::vector<double>& denom) {
		//mf_size_type m = v.size1();
		mf_size_type n = v.size2();
		mf_size_type r = w.size2();

		// compute numerator: t(w) [ v / w*h ]
		num.resize(r, n, false);
		std::fill(num.data().begin(), num.data().end(), 0.);
		const SparseMatrix::index_array_type& index1 = rowIndexData(v);
		const SparseMatrix::index_array_type& index2 = columnIndexData(v);
		const SparseMatrix::value_array_type& values = v.value_data();
		for (mf_size_type p=0; p<v.nnz(); p++) {
			mf_size_type i = index1[p];
			mf_size_type j = index2[p];
			double x = values[p];

            // compute the inner product
            double wh = 0;
            for (mf_size_type z=0; z<r; z++) {
                wh += w(i,z) * h(z,j);
            }

            // update numerator
            double f = x / wh;
            for (mf_size_type z=0; z<r; z++) {
                num(z,j) += w(i,z) * f;
            }
		}

		// compute denominator
		denom = sums2(w);

		// update h
	    for (mf_size_type z=0; z<r; z++) {
	        for (mf_size_type j=0; j<n; j++) {
	            h(z,j) *= num(z,j) / denom[z];
	        }
	    }
	}

	/** Updates w */
	void lee01Gkl_w(const SparseMatrix &v, DenseMatrix& w, const DenseMatrixCM& h,
			DenseMatrix& num, boost::numeric::ublas::vector<double>& denom) {
		mf_size_type m = v.size1();
		//mf_size_type n = v.size2();
		mf_size_type r = w.size2();

		// compute numerator: t(w) [ v / w*h ]
		num.resize(m, r, false);
		std::fill(num.data().begin(), num.data().end(), 0.);
		const SparseMatrix::index_array_type& index1 = rowIndexData(v);
		const SparseMatrix::index_array_type& index2 = columnIndexData(v);
		const SparseMatrix::value_array_type& values = v.value_data();
		for (mf_size_type p=0; p<v.nnz(); p++) {
			mf_size_type i = index1[p];
			mf_size_type j = index2[p];
			double x = values[p];

            // compute the inner product
            double wh = 0;
            for (mf_size_type z=0; z<r; z++) {
                wh += w(i,z) * h(z,j);
            }

            // update numerator
            double f = x / wh;
            for (mf_size_type z=0; z<r; z++) {
                num(i,z) += h(z,j) * f;
            }
		}

		// compute denominator
		denom = sums1(h);

		// update w
		for (mf_size_type i=0; i<m; i++) {
			for (mf_size_type z=0; z<r; z++) {
	            w(i,z) *= num(i,z) / denom[z];
	        }
	    }
	}

	/** Slower as temporaries are allocated */
	void lee01Gkl_w(const SparseMatrix &v, DenseMatrix& w, const DenseMatrixCM& h) {
		DenseMatrix num;
		boost::numeric::ublas::vector<double> denom;
		lee01Gkl_w(v, w, h, num, denom);
	}

	/** Slower as temporaries are allocated */
	void lee01Gkl_h(const SparseMatrixCM& v, const DenseMatrix& w, DenseMatrixCM& h) {
		DenseMatrixCM num;
		boost::numeric::ublas::vector<double> denom;
		lee01Gkl_h(v, w, h, num, denom);
	}
}

void lee01Gkl(FactorizationData<>& data, unsigned epochs, Trace& trace) {
	using namespace boost::numeric::ublas;
	LOG4CXX_INFO(detail::logger, "Starting Lee (2001) algorithm for GKL");
	NzslLoss testLoss; // will be used only if testData are provided

	// initialize
	double currentLoss=0;
	double timeLoss=0;

	rg::Timer t;
	t.start();
	currentLoss = gkl(data.v, data.w, data.h);
	t.stop();
	timeLoss=t.elapsedTime().nanos();
	LOG4CXX_INFO(mf::detail::logger, "Loss: " << currentLoss << " (" << t << ")");

	trace.clear();
	trace.add(new TraceEntry(currentLoss, timeLoss));

	// temporary variables
	DenseMatrix numW(data.w.size1(), data.w.size2());
	DenseMatrixCM numH(data.h.size1(), data.h.size2());
	boost::numeric::ublas::vector<double> denom;

	// main loop
	for (mf_size_type epoch=0; epoch<epochs; epoch++) {
		// run epoch
		t.start();
		if (epoch % 2 == 0) {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating W)");
			detail::lee01Gkl_w(data.v, data.w, data.h, numW, denom);
		} else {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating H)");
			detail::lee01Gkl_h(data.v, data.w, data.h, numH, denom);
		}
		t.stop();
		double timeEpoch = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Finished epoch " << (epoch+1) << " (" << t << ")");

		// compute loss
		t.start();
		currentLoss = gkl(data.v, data.w, data.h);
		t.stop();
		timeLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Loss: " << currentLoss << " (" << t << ")");

		// update trace
		trace.add(new TraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeEpoch, timeLoss));
	}

	LOG4CXX_INFO(detail::logger, "Finished Lee (2001) algorithm for GKL");
}

}
