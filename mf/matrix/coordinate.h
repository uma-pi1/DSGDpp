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
#ifndef MF_MATRIX_COORDINATE_H
#define MF_MATRIX_COORDINATE_H

#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace mf {

template<class D, std::size_t IB, class IA, class TA>
const IA& rowIndexData(const boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::row_major, IB, IA, TA> &m) {
	return m.index1_data();
}

template<class D, std::size_t IB, class IA, class TA>
const IA& rowIndexData(const boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::column_major, IB, IA, TA> &m) {
	return m.index2_data();
}

template<class D, std::size_t IB, class IA, class TA>
const IA& columnIndexData(const boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::row_major, IB, IA, TA> &m) {
	return m.index2_data();
}

template<class D, std::size_t IB, class IA, class TA>
const IA& columnIndexData(const boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::column_major, IB, IA, TA> &m) {
	return m.index1_data();
}


template<class D, std::size_t IB, class IA, class TA>
IA& rowIndexData(boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::row_major, IB, IA, TA> &m) {
	return m.index1_data();
}

template<class D, std::size_t IB, class IA, class TA>
IA& rowIndexData(boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::column_major, IB, IA, TA> &m) {
	return m.index2_data();
}

template<class D, std::size_t IB, class IA, class TA>
IA& columnIndexData(boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::row_major, IB, IA, TA> &m) {
	return m.index2_data();
}

template<class D, std::size_t IB, class IA, class TA>
IA& columnIndexData(boost::numeric::ublas::coordinate_matrix<D, boost::numeric::ublas::column_major, IB, IA, TA> &m) {
	return m.index1_data();
}


}

#endif
