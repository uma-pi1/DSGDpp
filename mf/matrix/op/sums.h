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
#ifndef MF_MATRIX_OP_SUMS_H
#define MF_MATRIX_OP_SUMS_H

#include <numeric>
#include <boost/foreach.hpp>

#include <mf/id.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/op/crossprod.h>

using namespace boost::numeric::ublas;
using namespace std;

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

/** Compute the row sums of the given matrix */
template<typename M>
boost::numeric::ublas::vector<typename M::value_type> sums1(M &m) {
	// TODO: slow (not reading entries in right order --> cache misses)

	typedef typename M::value_type T;
	mf_size_type n = m.size1();
	boost::numeric::ublas::vector<T> sums(n);
    for(mf_size_type i=0; i<n; i++) {
    	boost::numeric::ublas::matrix_row<M> row(m, i);
    	T s = T(boost::numeric::ublas::sum(row));
    	sums(i) = s;
    }
    return sums;
}

/** Compute the squared-entry row sums of the given matrix */
template<typename M>
boost::numeric::ublas::vector<typename M::value_type> squaredSums1(M &m) {
	// TODO: slow (not reading entries in right order --> cache misses)

	typedef typename M::value_type T;
	mf_size_type n = m.size1();
	boost::numeric::ublas::vector<T> squaredSums(n);
    for(mf_size_type i=0; i<n; i++) {
    	boost::numeric::ublas::matrix_row<M> row(m, i);
    	T s = 0;
    	BOOST_FOREACH(T v, row) {
    		s += v*v;
    	}
    	squaredSums[i] = s;
    }
    return squaredSums;
}
/** Compute the squared-entry row sums of the given matrix divided by the row or column sums
 * 	make sure that nnz.size()=m.size2()
 */
template<typename M>
boost::numeric::ublas::vector<typename M::value_type> nzl2SquaredSums1(M &m,const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	typedef typename M::value_type T;
	boost::numeric::ublas::vector<T> squaredSums( m.size1(), 0 );

	for (mf_size_type j=0;j<m.size2(); j++) {
    	mf_size_type nnzj = nnz[j + nnzOffset];
    	for(mf_size_type i=0; i<m.size1(); i++) {
    		double v = m(i,j);
    		squaredSums[i] += v*v*nnzj;
    	}
    }
    return squaredSums;
}

/** Compute the column sums of the given matrix */
template<typename M>
boost::numeric::ublas::vector<typename M::value_type> sums2(M &m) {
	// TODO: slow (not reading entries in right order --> cache misses)

	typedef typename M::value_type T;
	mf_size_type n = m.size2();
	boost::numeric::ublas::vector<T> sums(n);
    for(mf_size_type i=0; i<n; i++) {
    	boost::numeric::ublas::matrix_column<M> col(m, i);
    	T s = T(boost::numeric::ublas::sum(col));
    	sums(i) = s;
    }
    return sums;
}

/** Compute the squared-entry column sums of the given matrix divided by the row or column sums
 * 	make sure that nnz.size()=m.size1()
 */
template<typename M>
boost::numeric::ublas::vector<typename M::value_type> nzl2SquaredSums2(M &m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	typedef typename M::value_type T;

	boost::numeric::ublas::vector<T> squaredSums( m.size2(), 0 );
	for (mf_size_type i=0;i<m.size1();i++){
    	mf_size_type nnzi = nnz[i + nnzOffset];
    	for(mf_size_type j=0; j<m.size2(); j++) {
    		double v = m(i,j);
    		squaredSums[j] += v*v*nnzi;
    	}
    }
    return squaredSums;
}

/** Compute the squared-entry column sums of the given matrix */
template<typename M>
boost::numeric::ublas::vector<typename M::value_type> squaredSums2(M &m) {
	// TODO: slow (not reading entries in right order --> cache misses)
	typedef typename M::value_type T;
	mf_size_type n = m.size2();
	boost::numeric::ublas::vector<T> squaredSums(n);
    for(mf_size_type i=0; i<n; i++) {
    	boost::numeric::ublas::matrix_column<M> col(m, i);
    	T s = 0;
    	BOOST_FOREACH(T v, col) {
    		s += v*v;
    	}
    	squaredSums[i] = s;
    }
    return squaredSums;
}

/** Compute the sum of vectors stored in the given matrix*/
template<typename M>
M vectorSum(boost::numeric::ublas::matrix<M>& m){
	unsigned r = m(0,0).size();
	unsigned mm = m.size1();
	unsigned nn = m.size2();
	M result(r);
	for (unsigned i=0; i<result.size(); i++) {
		result(i) = 0;
	}
	for (unsigned i=0; i<mm; i++) {
		for (unsigned j=0; j<nn; j++) {
			result += m(i,j);
		}
	}
	return result;
}

/** For given matrices W and H compute sum of squares of W %*% H */
inline double sumOfSquares(const DenseMatrix& w, const DenseMatrixCM& h) {
	BOOST_ASSERT( w.size2() == h.size1() );
	mf_size_type r = w.size2();
	DenseMatrixCM hht = tcrossprod(h);
	DenseMatrix wtw = crossprod(w);

	double result = 0;
	for (mf_size_type i=0; i<r; i++) {
		for (mf_size_type j=0; j<r; j++) {
			result += wtw(i,j) * hht(i,j);
		}
	}
	return result;
}


// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	template<typename M>
	struct Sums1Task : public PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, sums1<M>, ID_SUMS1> {
		typedef PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, sums1<M>, ID_SUMS1> Task;
	};

	template<typename M>
	struct Sums2Task : public PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, sums2<M>, ID_SUMS2> {
		typedef PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, sums2<M>, ID_SUMS2> Task;
	};

	template<typename M>
	struct SquaredSums1Task : public PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, squaredSums1<M>, ID_SQUARED_SUMS1> {
		typedef PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, squaredSums1<M>, ID_SQUARED_SUMS1> Task;
	};

	template<typename M>
	struct SquaredSums2Task : public PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, squaredSums2<M>, ID_SQUARED_SUMS2> {
		typedef PerBlockTaskReturn<M, boost::numeric::ublas::vector<typename M::value_type>, squaredSums2<M>, ID_SQUARED_SUMS2> Task;
	};
}

template<typename M>
inline void sums1(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> >& result, int tasksPerRank=1) {
	runTaskOnBlocks< detail::Sums1Task<M> >(m, result, tasksPerRank);
}

template<typename M>
inline boost::numeric::ublas::vector<typename M::value_type> sums1(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	BOOST_ASSERT( m.blocks1() == 1);
	boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> > sums1result(m.blocks1(), m.blocks2());
	sums1(m, sums1result, tasksPerRank);
	return vectorSum(sums1result);
}

template<typename M>
inline void sums2(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> >& result, int tasksPerRank=1) {
	runTaskOnBlocks< detail::Sums2Task<M> >(m, result, tasksPerRank);
}

template<typename M>
inline boost::numeric::ublas::vector<typename M::value_type> sums2(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	BOOST_ASSERT( m.blocks2() == 1);
	boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> > sums2result(m.blocks1(), m.blocks2());
	sums2(m, sums2result, tasksPerRank);
	return vectorSum(sums2result);
}

template<typename M>
inline void squaredSums1(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> >& result, int tasksPerRank=1) {
	runTaskOnBlocks< detail::SquaredSums1Task<M> >(m, result, tasksPerRank);
}

template<typename M>
inline boost::numeric::ublas::vector<typename M::value_type> squaredSums1(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	BOOST_ASSERT( m.blocks1() == 1);
	boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> > sums1result(m.blocks1(), m.blocks2());
	squaredSums1(m, sums1result, tasksPerRank);
	return vectorSum(sums1result);
}

template<typename M>
inline void squaredSums2(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> >& result, int tasksPerRank=1) {
	runTaskOnBlocks< detail::SquaredSums2Task<M> >(m, result, tasksPerRank);
}

template<typename M>
inline boost::numeric::ublas::vector<typename M::value_type> squaredSums2(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	BOOST_ASSERT( m.blocks2() == 1);
	boost::numeric::ublas::matrix<boost::numeric::ublas::vector<typename M::value_type> > sums2result(m.blocks1(), m.blocks2());
	squaredSums2(m, sums2result, tasksPerRank);
	return vectorSum(sums2result);
}

/** For given matrices W and H compute sum of squares of W %*% H */
inline double sumOfSquares(const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h, int tasksPerRank = 1) {
	BOOST_ASSERT( w.size2() == h.size1() );
	mf_size_type r = w.size2();

	DenseMatrixCM hht = tcrossprod(h, tasksPerRank);
	DenseMatrix wtw = crossprod(w, tasksPerRank);

	double result = 0;
	for (mf_size_type i=0; i<r; i++) {
		for (mf_size_type j=0; j<r; j++) {
			result += wtw(i,j) * hht(i,j);
		}
	}
	return result;
}


namespace detail {

/**	A task for calculating nzl2SquaredSums1 and szl2SquaredSums2 */
struct Nzl2SquaredSumsTask {
	/**
	 * 	The argument that is necessary for a task Nzl2SquaredSumsTask. Described in terms of:
	 * 	(1) the block of W or H on which the task will operate
	 * 	(2) the vector of sums of nnz entries the row or column for W or H respectively
	 * 	(3) the start and end indices defining the range of the nnz vector which correspond to the specific block
	 * 	(4) the information about if the input block belongs to W (row=true) or H (row=false)
	 */
	struct Arg {
	public:
		Arg() : data(mpi2::UNINITIALIZED) {};

		Arg(mpi2::RemoteVar block, const std::string& nnzName, mf_size_type nnzOffset, bool isRowFactor)
		: data(block), nnzName(nnzName), nnzOffset(nnzOffset), isRowFactor(isRowFactor){}

		static Arg constructArgW(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName) {
			return Arg(block, nnzName, m.blockOffset1(b1), true);
		}

		static Arg constructArgH(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName) {
			return Arg(block, nnzName, m.blockOffset2(b2), false);
		}

		mpi2::RemoteVar data;
		std::string nnzName;
		mf_size_type nnzOffset;
		bool isRowFactor;

	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & data;
			ar & nnzName;
			ar & nnzOffset;
			ar & isRowFactor;
		}
	};

	static const std::string id() {	return std::string("__mf/matrix/op/Nzl2SquaredSumsTask") ;}

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> args;
		ch.recv(args);
		std::vector<boost::mpi::request> reqs(args.size());
		boost::numeric::ublas::vector<boost::numeric::ublas::vector<double> > results(args.size());
		for (unsigned i=0; i<args.size(); i++) {
			Arg& arg = args[i];
			const std::vector<mf_size_type>& nnz = *mpi2::env().get<std::vector<mf_size_type> >(arg.nnzName);
			if (arg.isRowFactor){
				const DenseMatrix& m = *arg.data.getLocal<DenseMatrix>();
				results[i] = nzl2SquaredSums2(m, nnz, arg.nnzOffset);
			}
			else{
				const DenseMatrixCM& m = *arg.data.getLocal<DenseMatrixCM>();
				results[i] = nzl2SquaredSums1(m, nnz, arg.nnzOffset);
			}
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

} // namespace detail



inline boost::numeric::ublas::vector<double> nzl2SquaredSums2(const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName, int tasksPerRank) {
	BOOST_ASSERT( m.blocks2() == 1);
	boost::numeric::ublas::matrix<boost::numeric::ublas::vector<double> > nzl2SquaredSums2result(m.blocks1(), m.blocks2());
	runTaskOnBlocks<DenseMatrix, boost::numeric::ublas::vector<double>, detail::Nzl2SquaredSumsTask::Arg>(
			m,
			nzl2SquaredSums2result,
			boost::bind(detail::Nzl2SquaredSumsTask::Arg::constructArgW, _1, _2, _3, boost::cref(m), boost::cref(nnzName)),
			detail::Nzl2SquaredSumsTask::id(),
			tasksPerRank,
			false);
	return vectorSum(nzl2SquaredSums2result);
}

inline boost::numeric::ublas::vector<double> nzl2SquaredSums1(const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName, int tasksPerRank) {
	BOOST_ASSERT( m.blocks1() == 1);
	boost::numeric::ublas::matrix<boost::numeric::ublas::vector<double> > nzl2SquaredSums1result(m.blocks1(), m.blocks2());
	runTaskOnBlocks<DenseMatrixCM, boost::numeric::ublas::vector<double>, detail::Nzl2SquaredSumsTask::Arg>(
			m,
			nzl2SquaredSums1result,
			boost::bind(detail::Nzl2SquaredSumsTask::Arg::constructArgH, _1, _2, _3, boost::cref(m), boost::cref(nnzName)),
			detail::Nzl2SquaredSumsTask::id(),
			tasksPerRank,
			false);
	return vectorSum(nzl2SquaredSums1result);
}


}

#endif
