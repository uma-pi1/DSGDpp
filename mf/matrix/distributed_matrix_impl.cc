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
#include <mf/matrix/distributed_matrix.h>

namespace mf {

std::string defaultBlockName(const std::string& name, mf_size_type b1, mf_size_type b2) {
	std::stringstream ss;
	ss << name << "/block(" << b1 << "," << b2 << ")";
	return ss.str();
}

void computeDefaultBlockOffsets(mf_size_type size, mf_size_type blocks,
		std::vector<SparseMatrix::size_type>& blockOffsets) {
	blockOffsets.resize(blocks);
	mf_size_type minSize = size/blocks;
	mf_size_type remainder = size % blocks;
	for (mf_size_type i=0; i<blocks; i++) {
		if (i == 0) {
			blockOffsets[i] = 0;
		} else {
			blockOffsets[i] = minSize + blockOffsets[i-1];
			if (remainder > 0) {
				++blockOffsets[i];
				--remainder;
			}
		}
	}
};

void computeDefaultBlockLocations(unsigned worldSize,
		mf_size_type blocks1, mf_size_type blocks2,
		bool partitionByRow, boost::numeric::ublas::matrix<int>& blockLocations) {
	blockLocations.resize(blocks1, blocks2, false);
	if (partitionByRow) {
		int i = 0;
		for (unsigned p=0; p<worldSize; p++) {
			mf_size_type blocksAtNode = blocks1/worldSize + (p < blocks1 % worldSize ? 1 : 0);
			for (mf_size_type k=0; k<blocksAtNode; k++, i++) {
				for (mf_size_type j=0; j<blocks2; j++){
					blockLocations(i,j) = p;
				}
			}
		}
	} else { // !partitionByRow
		int j = 0;
		for (unsigned p=0; p<worldSize; p++) {
			mf_size_type blocksAtNode = blocks2/worldSize + (p < blocks2 % worldSize ? 1 : 0);
			for (mf_size_type k=0; k<blocksAtNode; k++, j++) {
				for (mf_size_type i=0; i<blocks1; i++){
					blockLocations(i,j) = p;
				}
			}
		}
	}
}

void computeDefaultBlockVars(
		const std::string& name,
		const boost::numeric::ublas::matrix<int>& blockLocations,
		boost::numeric::ublas::matrix<mpi2::RemoteVar>& blockVars) {
	mf_size_type blocks1 = blockLocations.size1();
	mf_size_type blocks2 = blockLocations.size2();
	blockVars.resize(blocks1, blocks2, false);
	for (mf_size_type b1=0; b1<blocks1; b1++) {
		for (mf_size_type b2=0; b2<blocks2; b2++) {
			blockVars(b1,b2) = mpi2::RemoteVar(
					blockLocations(b1,b2),
					defaultBlockName(name, b1, b2)
					);
		}
	}
}


} // namepsace mf

// externed stuff
/*
namespace boost { namespace numeric { namespace ublas {
template class coordinate_matrix<double, boost::numeric::ublas::row_major>;
template class coordinate_matrix<double, boost::numeric::ublas::column_major>;
template class boost::numeric::ublas::matrix<int, boost::numeric::ublas::row_major>;
template class boost::numeric::ublas::matrix<unsigned, boost::numeric::ublas::row_major>;
template class boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major>;
template class boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>;
}}}

namespace mf {
template class DistributedMatrix<SparseMatrix>;
template class DistributedMatrix<SparseMatrixCM>;
template class DistributedMatrix<DenseMatrix>;
template class DistributedMatrix<DenseMatrixCM>;
}
*/
