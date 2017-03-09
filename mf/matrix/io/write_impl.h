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
#include <mf/matrix/io/write.h>   // compiler hint

#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

#include <util/exception.h>
#include <util/io.h>

#include <mf/matrix/coordinate.h>

namespace mf {

namespace detail {

template<class Matrix>
void writeMmCoord(const std::string& fname, const Matrix& M) {
	RG_THROW(rg::NotImplementedException, "writing non-coordinate matrices to MatrixMarket coordinate format");
}

template<typename L, std::size_t IB, class IA, class TA>
void writeMmCoord(const std::string& fname, const boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA> &M) {
	// open file
	std::ofstream out(fname.c_str());
	if (!out.is_open())
		RG_THROW(rg::IOException, std::string("Cannot open file ") + fname);
	out.precision(std::numeric_limits<double>::digits10 + 1);


	// write header
	out << "%%MatrixMarket matrix coordinate real general" << std::endl;
	out << "%=================================================================================" << std::endl;
	out << "%" << std::endl;
	out << "% This ASCII file represents a sparse MxN matrix with L" << std::endl;
	out << "% nonzeros in the following Matrix Market format:" << std::endl;
	out << "%" << std::endl;
	out << "% +----------------------------------------------+" << std::endl;
	out << "% |%%MatrixMarket matrix coordinate real general | <--- header line" << std::endl;
	out << "% |%                                             | <--+" << std::endl;
	out << "% |% comments                                    |    |-- 0 or more comment lines" << std::endl;
	out << "% |%                                             | <--+" << std::endl;
	out << "% |    M  N  L                                   | <--- rows, columns, entries" << std::endl;
	out << "% |    I1  J1  A(I1, J1)                         | <--+" << std::endl;
	out << "% |    I2  J2  A(I2, J2)                         |    |" << std::endl;
	out << "% |    I3  J3  A(I3, J3)                         |    |-- L lines" << std::endl;
	out << "% |        . . .                                 |    |" << std::endl;
	out << "% |    IL JL  A(IL, JL)                          | <--+" << std::endl;
	out << "% +----------------------------------------------+   " << std::endl;
	out << "%" << std::endl;
	out << "% Indices are 1-based, i.e. A(1,1) is the first element." << std::endl;
	out << "%" << std::endl;
	out << "%=================================================================================" << std::endl;

	// write dimension line
	out << M.size1() << " " << M.size2() << " " << M.nnz() << std::endl;
	std::cout<<fname.c_str()<<"nnz: "<<M.nnz()<<std::endl;

	// write matrix
	const IA &is = rowIndexData(M);
	const IA &js = columnIndexData(M);
	const TA &xs = M.value_data();
	for (mf_size_type i=0; i<M.nnz(); i++) {
		out << (is[i]+1) << " " << (js[i]+1) << " " << xs[i] << std::endl;
	}

	// done
	out.close();
}


template<class Matrix>
void writeMmArray(const std::string& fname, const Matrix& M) {
	// open file
	std::ofstream out(fname.c_str());
	if (!out.is_open())
		RG_THROW(rg::IOException, std::string("Cannot open file ") + fname);
	out.precision(std::numeric_limits<double>::digits10 + 1);


	// write header
	out << "%%MatrixMarket matrix array real general" << std::endl;
	out << "% First line: ROWS COLUMNS" << std::endl;
	out << "% Subsequent lines: entries in column-major order" << std::endl;

	// write dimension line
	out << M.size1() << " " << M.size2() << std::endl;
	
	// write matrix
	for (mf_size_type j=0; j<M.size2(); j++) {
		for (mf_size_type i=0; i<M.size1(); i++) {
			 out << M(i,j) << std::endl;
		}
	}

	// done
	out.close();
}

template<class Matrix>
void writeBoostText(const std::string& fname, const Matrix& M) {
	std::ofstream out(fname.c_str());
	boost::archive::text_oarchive oa(out);
	oa << M;
}

template<class Matrix>
void writeBoostBin(const std::string& fname, const Matrix& M) {
	std::ofstream out(fname.c_str());
	boost::archive::binary_oarchive oa(out);
	oa << M;
}

} // namespace detail

template<typename M>
void writeMatrix(const std::string& fname, const M& m, MatrixFileFormat format) {
	if (format == AUTOMATIC) {
		format = getMatrixFormat(fname);
	}
	switch (format) {
	case AUTOMATIC: // handled above
	case MM_ARRAY:
		detail::writeMmArray(fname, m);
		break;
	case MM_COORD:
		detail::writeMmCoord(fname, m);
		break;
	case BOOST_SPARSE_BIN:
	case BOOST_DENSE_BIN:
		detail::writeBoostBin(fname, m);
		break;
	case BOOST_SPARSE_TEXT:
	case BOOST_DENSE_TEXT:
		detail::writeBoostText(fname, m);
		break;
	default:
		RG_THROW(rg::InvalidArgumentException, "invalid matrix format");
	}
}

} // namespace mf

