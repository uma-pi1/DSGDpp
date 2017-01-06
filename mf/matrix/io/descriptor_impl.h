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
#include <mf/matrix/io/descriptor.h>   // compiler hint

namespace mf {

namespace detail {
	template<class M>
	BlockedMatrixFileDescriptor blockedMatrixFileDescriptorCreate(
		const DistributedMatrix<M>& m,
		const std::string& path,
		const std::string& baseFilename, MatrixFileFormat format) {
		BlockedMatrixFileDescriptor result;
		result.size1 = m.size1();
		result.size2 = m.size2();
		result.blocks1 = m.blocks1();
		result.blocks2 = m.blocks2();
		result.blockOffsets1 = m.blockOffsets1();
		result.blockOffsets2 = m.blockOffsets2();
		result.path = path;
		result.filenames = boost::numeric::ublas::matrix<std::string>(m.blocks1(), m.blocks2());
		result.format = format;
		std::string extension = getExtension(result.format);
		for (mf_size_type bi = 0; bi<result.blocks1; bi++) {
			for (mf_size_type bj = 0; bj<result.blocks2; bj++) {
				std::stringstream ss;
				ss <<  baseFilename << "-" << bi << "-" << bj << "." << extension;
				result.filenames(bi,bj) = ss.str();
			}
		}
		return result;
	}
}

template<class T, class L, class A>
BlockedMatrixFileDescriptor BlockedMatrixFileDescriptor::create(
		const DistributedMatrix<boost::numeric::ublas::matrix<T,L,A> >& m,
		const std::string& path,
		const std::string& baseFilename, MatrixFileFormat format) {
	return detail::blockedMatrixFileDescriptorCreate(m, path, baseFilename, format);
};

template<class T, class L, std::size_t IB, class IA, class TA>
BlockedMatrixFileDescriptor BlockedMatrixFileDescriptor::create(
		const DistributedMatrix<boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA> >& m,
		const std::string& path,
		const std::string& baseFilename, MatrixFileFormat format) {
	return detail::blockedMatrixFileDescriptorCreate(m, path, baseFilename, format);
};

}
