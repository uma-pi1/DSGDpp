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
#ifndef MF_MATRIX_SHUFFLE_H
#define MF_MATRIX_SHUFFLE_H

#include <util/random.h>


namespace mf {

/** Creates a random permutation matrix of the specified size */
inline void generateRandomPermutationMatrix(
		boost::numeric::ublas::coordinate_matrix<mf_size_type>& m,
		mf_size_type size,
		rg::Random32& random) {
	m.resize(size, size, false);
	m.clear();
	m.reserve(size);

	std::vector<mf_size_type> v(m.size1());
	for (mf_size_type i=0; i<m.size1(); i++) v[i] = i;
	rg::shuffle(v.begin(), v.end(), random);
	for (mf_size_type i=0; i<m.size1(); i++) m.push_back(i, v[i], 1);
}

/** Shuffles the rows and columns of the given matrix */
template<typename T, typename L, typename A>
void shuffle(boost::numeric::ublas::matrix<T, L, A>& m, rg::Random32& random) {
	// create a permutation matrix for rows and columns
	boost::numeric::ublas::coordinate_matrix<mf_size_type> r(m.size1(), m.size1());
	generateRandomPermutationMatrix(r, m.size1(), random);
	boost::numeric::ublas::coordinate_matrix<mf_size_type> c(m.size2(), m.size2());
	generateRandomPermutationMatrix(c, m.size2(), random);

	// apply the permutation
	m = boost::numeric::ublas::prod(r, m);
	m = boost::numeric::ublas::prod(m, c);
}

}

#endif
