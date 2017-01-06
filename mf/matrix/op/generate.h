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
/** \file
 *
 *  Methods to generate (random) matrices.
 */

#ifndef MF_MATRIX_GENERATE_H
#define MF_MATRIX_GENERATE_H

#include <boost/random.hpp>
#include <boost/random/poisson_distribution.hpp>

#include <util/random.h>

#include <mf/matrix/coordinate.h>

namespace mf {
/** Apply a function to each element of a submatrix of a dense matrix.
 *
 * @param[in,out] m a matrix
 * @param function to apply to each element; double f(mf_size_type i, mf_size_type j, double x)
 * @tparam F type of function
 */
template<typename L, typename A, typename F>
void apply(boost::numeric::ublas::matrix<double, L, A>& m,
		F f,
		mf_size_type start1, mf_size_type end1,
		mf_size_type start2, mf_size_type end2) {

	for (mf_size_type i=start1; i<end1; i++) {
		for (mf_size_type j=start2; j<end2; j++) {
			m(i,j) = f(i-start1, j-start2, m(i,j));
		}
	}
};

/** Apply a function to each nonzero element of a sparse matrix.
 *
 * @param[in,out] m a matrix
 * @param function to apply to each element; double f(mf_size_type i, mf_size_type j, double x)
 * @tparam F type of function
 */
template<class T, class L, std::size_t IB, class IA, class TA,typename F>
void apply(boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>& m, F f,
		mf_size_type nnz, mf_size_type nnzStart,
		mf_size_type start1,mf_size_type end1, mf_size_type start2,  mf_size_type end2) {
	IA& index1 = rowIndexData(m);
	IA& index2 = columnIndexData(m);
	TA& values = m.value_data();

	//size of m
	mf_size_type size1=end1-start1;
	mf_size_type size2=end2-start2;

	if (nnz>size1*size2) {
		RG_THROW(rg::InvalidArgumentException, "nnz must be less than matrix size");
	}

	for (mf_size_type p=nnzStart;p<nnzStart+nnz;p++){
		if (index2[p]<start2 || index2[p]>end2)//still scanning irrelevant entries
			continue;
		values[p] = f(index1[p]-start1, index2[p]-start2, values[p]);
	}

};

namespace detail {
template<typename F>
double applyAdd(mf_size_type i, mf_size_type j, double v, F& f) {
	return v + f();
}
}

/** Add a random number to each element of a dense matrix / each nonzero element of a sparse matrix.
 *
 * @param[in,out] m matrix to which to add noise
 * @param f a Boost distribution (e.g., boost::uniform_real)
 * @tparam M matrix type
 * @tparam Dist Boost distribution type
 */
template<typename M, typename Dist>
void addRandom(M& m, rg::Random32& random, Dist f) {
	typedef boost::variate_generator<rg::Random32::Prng&, Dist> Gen;
	Gen gen(random.prng(), f);
	apply(m,
			boost::bind(
					detail::applyAdd<Gen>,
					_1, _2, _3,
					boost::ref(gen)
			),m.nnz(),0,0,m.size1(),0,m.size2()///////
	);
}
template<typename M, typename Dist>
void addRandom(M& m, rg::Random32& random, Dist f, mf_size_type start1,mf_size_type end1,mf_size_type start2,  mf_size_type end2) {
	typedef boost::variate_generator<rg::Random32::Prng&, Dist> Gen;
	Gen gen(random.prng(), f);
	apply(m,
			boost::bind(
					detail::applyAdd<Gen>,
					_1, _2, _3,
					boost::ref(gen)
			),start1,end1,start2,end2
	);
}
template<typename M, typename Dist>
void addRandom(M& m, rg::Random32& random, Dist f,mf_size_type nnz,mf_size_type nnzStart, mf_size_type start1,mf_size_type end1,mf_size_type start2,  mf_size_type end2) {
	typedef boost::variate_generator<rg::Random32::Prng&, Dist> Gen;
	Gen gen(random.prng(), f);
	apply(m,
			boost::bind(
					detail::applyAdd<Gen>,
					_1, _2, _3,
					boost::ref(gen)
			),nnz,nnzStart, start1,end1,start2,end2
	);
}

namespace detail {
inline double applyPoisson(mf_size_type i, mf_size_type j, double v, rg::Random32& random) {
	typedef boost::poisson_distribution<mf_size_type,double> Dist;
	typedef boost::variate_generator<rg::Random32::Prng&,Dist> Gen;
	Dist dist(v);
	Gen gen(random.prng(), dist);
	return gen();
}
}

/** Replaces each entry of a dense matrix / each nonzero entry of a sparse matrix by a random
 * Poisson with mean equal to the entry.
 *
 * @param[in,out] m matrix to which to add noise
 * @tparam M matrix type
 */
template<typename L,typename A>
void applyPoisson(boost::numeric::ublas::matrix<double, L, A>& m, rg::Random32& random) {
	apply(m,
			boost::bind(
					detail::applyPoisson,
					_1, _2, _3,
					boost::ref(random)
			),0,m.size1(),0,m.size2()
	);
}

template<class T, class L, std::size_t IB, class IA, class TA>
void applyPoisson(boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>& m, rg::Random32& random) {
	apply(m,
			boost::bind(
					detail::applyPoisson,
					_1, _2, _3,
					boost::ref(random)
			),m.nnz(), 0,0,m.size1(), 0,  m.size2()
	);
}

/** Generate a random i.i.d. dense matrix using a Boost distribution.
 *
 * @param[in,out] m matrix to be filled with random values
 * @param f a Boost distribution (e.g., boost::uniform_real)
 * @tparam M dense matrix type
 * @tparam Dist Boost distribution type
 */
template<typename M, typename Dist>
void generateRandom(M& m, rg::Random32& random, Dist f,
		mf_size_type start1, mf_size_type end1, mf_size_type start2, mf_size_type end2) {
	typedef boost::variate_generator<rg::Random32::Prng&, Dist> Gen;
	Gen gen(random.prng(), f);
	apply(m,
			boost::bind(
					static_cast<double (Gen::*)()>(&Gen::operator()),
					boost::ref(gen)
			), start1, end1, start2, end2
	);
}

template<typename M, typename Dist>
void generateRandom(M& m, rg::Random32& random, Dist f) {
	generateRandom(m, random, f, 0, m.size1(), 0, m.size2());
}

/** Generate a dense matrix by selecting random positions and computing the value at these positions.
 *
 * @param random a pseudo-random number generator
 * @param[in,out] output matrix (size will be retained)
 * @param nnz number of nonzero elements to select
 * @param function to apply to compute a selected element; double f(mf_size_type i, mf_size_type j)
 * @tparam M matrix type
 * @tparam F function type
 *
 */
template<typename M, typename F>
void generate(rg::Random32& random, M& m, mf_size_type nnz, F f,
		mf_size_type start1,mf_size_type end1, mf_size_type start2,mf_size_type end2) {
	mf_size_type size1=end1-start1;
	mf_size_type size2=end2-start2;

	if (nnz>m.size1()*m.size2()) {
		RG_THROW(rg::InvalidArgumentException, "nnz must be less than matrix size");
	}

	// clear the data matrix
	m.clear();

	// generate
	mf_size_type N = size1*size2;
	mf_size_type current = 0;
	while (nnz > 0) {
		mf_size_type skip = rg::skipSequential(random, nnz, N-current);
		current += skip;
		mf_size_type i = current / size2;
		mf_size_type j = current % size2;
		m(start1+i, start2+j) = f(i,j);// is this correct?
		nnz--;
		current++;
	}
}

template<typename F>
void generate(rg::Random32& random, DenseMatrix& m, mf_size_type nnz, F f) {
	generate<DenseMatrix,F>(random, m, nnz, f, 0,  m.size1(),0,m.size2());
}
template<typename F>
void generate(rg::Random32& random, DenseMatrixCM& m, mf_size_type nnz, F f) {
	generate<DenseMatrixCM,F>(random, m, nnz, f, 0, m.size1(),0,m.size2());
}



/** Generate a sparse matrix by selecting random positions and computing the value at these positions.
 *
this is called for generating matrices on the fly in distributed manner
H might be much bigger than m. HOFFSET specifies to which column of h the first column of m corresponds
I hope that this should work the same for CM
 */
template<class T, class L, std::size_t IB, class IA, class TA,typename F>
void generate(rg::Random32& random, boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>& m, mf_size_type nnz,
		F f, mf_size_type nnzStart, mf_size_type hOffset,
		mf_size_type start1, mf_size_type end1, mf_size_type start2, mf_size_type end2) {
	//size of m
	mf_size_type size1 = end1-start1;
	mf_size_type size2 = end2-start2;

	IA& index1 = rowIndexData(m);
	IA& index2 = columnIndexData(m);
	TA& values = m.value_data();

	if (nnz>size1*size2) {
		RG_THROW(rg::InvalidArgumentException, "nnz must be less than matrix size");
	}

	// generate
	mf_size_type N = size1*size2;

	mf_size_type current = 0;
	mf_size_type ind = nnzStart;
	while (nnz > 0) {
		mf_size_type skip = rg::skipSequential(random, nnz, N-current);
		current += skip;

		mf_size_type i = current / size2;
		mf_size_type j = current % size2;

		index1[ind] = start1+i;
		index2[ind] = start2+j;
		values[ind] = f(start1+i, start2+hOffset+j);
		//std::cout<<"index1: "<<index1[nnzStart+ind]<<" index2: "<<index2[nnzStart+ind]<<" values: "<<values[nnzStart+ind]<<std::endl;
		nnz--;
		current++;
		ind++;
	}
}

//this is the original declaration
template<typename F>
void generate(rg::Random32& random, SparseMatrix& m, mf_size_type nnz, F f) {
	if (nnz>m.size1()*m.size2()) {
		RG_THROW(rg::InvalidArgumentException, "nnz must be less than matrix size");
	}

	// clear the data matrix
	m.clear();
	m.reserve(nnz, false);

	// generate
	mf_size_type N = m.size1()*m.size2();
	mf_size_type current = 0;
	while (nnz > 0) {
		mf_size_type skip = rg::skipSequential(random, nnz, N-current);
		current += skip;
		mf_size_type i = current / m.size2();
		mf_size_type j = current % m.size2();
		m.append_element(i, j, f(i,j));
		nnz--;
		current++;
	}
}



/** Generate a random i.i.d. sparse matrix using a Boost distribution.
 *
 * @param[in,out] output matrix (size will be retained)
 * @param nnz number of nonzero elements to select
 * @param random a pseudo-random number generator
 * @param f a Boost distribution (e.g., boost::uniform_real) *
 * @tparam M dense matrix type // perhaps sparse?
 * @tparam Dist Boost distribution type
 */
template<typename M, typename Dist>
void generateRandom(M& m, mf_size_type nnz, rg::Random32& random, Dist f) {
	typedef boost::variate_generator<rg::Random32::Prng&, Dist> Gen;
	Gen gen(random.prng(), f);
	generate(random, m, nnz,
			boost::bind(
					static_cast<double (Gen::*)()>(&Gen::operator()),
					boost::ref(gen)
			), 0, m.size1(), 0, m.size2()
	);
}

namespace detail {
template<typename M2, typename M3>
double innerProduct(const M2& w, const M3& h, mf_size_type i, mf_size_type j) {
	double result = 0;
	for (mf_size_type k=0; k<w.size2(); k++) {
		result += w(i,k)*h(k,j);
	}
	return result;
}
}

/** Generates a random sparse matrix with entries from factors.
 *
 * @param[out] v the result matrix
 * @param nnz number of non-zero entries to sample
 * @param w matrix of row factors
 * @param h matrix of column factors
 * @param random a pseudo-random number generator
 *
 * @tparam M1 dense matrix type of output matrix
 * @tparam M2 dense matrix type of row factors
 * @tparam M3 dense matrix type of column factors
 */
template<typename M1, typename M2, typename M3>
void generateRandom(M1& v, mf_size_type nnz, const M2& w, const M3& h, rg::Random32& random) {
	v.resize(w.size1(), h.size2(), false);
	generate(random, v, nnz, boost::bind(
			detail::innerProduct<M2,M3>,
			boost::cref(w), boost::cref(h), _1, _2
	));
}

template<typename M1, typename M2, typename M3>
void generateRandom(M1& v, mf_size_type nnz, const M2& w, const M3& h, rg::Random32& random,mf_size_type nnzStart, mf_size_type hOffset,
		mf_size_type start1, mf_size_type end1, mf_size_type start2, mf_size_type end2) {
	generate(random, v, nnz, boost::bind(
			detail::innerProduct<M2,M3>,
			boost::cref(w), boost::cref(h), _1, _2
	), nnzStart, hOffset, start1, end1, start2, end2
	);
}



}

#endif
