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
#include <mf/matrix/distribute.h>

namespace mf {

namespace detail {

void assignBlocksToTasks(
		const boost::numeric::ublas::matrix<mpi2::RemoteVar>& blocks,
		int ranks, int tasksPerRank,
		boost::numeric::ublas::matrix<int>& groupIds
) {
	const mf_size_type blocks1 = blocks.size1();
	const mf_size_type blocks2 = blocks.size2();

	// figure out which rank has which blocks
	std::vector<std::pair<mf_size_type,mf_size_type> > blocksForRank[ranks];
        // variant 1: spread blocks
        //	for (mf_size_type b2offset=0; b2offset < tasksPerRank; b2offset++) { // order is important!
        //		for (mf_size_type b2base=0; b2base<blocks2; b2base+=tasksPerRank) {
        //			mf_size_type b2 = b2base + b2offset;
        //			if (b2 >= blocks2) break;
	//		for (mf_size_type b1=0; b1<blocks1; b1++) {
	//			int rank = blocks(b1,b2).rank();
	//			blocksForRank[rank].push_back(std::pair<mf_size_type,mf_size_type>(b1,b2));
	//		}
        //        }
        // }
        // variant 2: group blocks
        for (mf_size_type b2=0; b2<blocks2; b2++) {
            for (mf_size_type b1=0; b1<blocks1; b1++) {
                int rank = blocks(b1,b2).rank();
                blocksForRank[rank].push_back(std::pair<mf_size_type,mf_size_type>(b1,b2));
            }
        }

	// assign blocks to tasks
	groupIds.resize(blocks1, blocks2, false);
	for (int rank=0; rank<ranks; rank++) {
		mf_size_type n = blocksForRank[rank].size();
		mf_size_type start = 0;
		for (unsigned task=0; task<(unsigned)tasksPerRank; task++) {
			mf_size_type end = start + n/tasksPerRank + (task < n % tasksPerRank ? 1 : 0);
			for (mf_size_type k=start; k<end; k++) {
				mf_size_type b1 = blocksForRank[rank][k].first;
				mf_size_type b2 = blocksForRank[rank][k].second;
				groupIds(b1,b2) = rank*tasksPerRank + task;
			}
			start = end;
		}
	}

	LOG4CXX_DEBUG(logger, "Group ids: " << groupIds);
}

}

template void runTaskOnBlocks3<SparseMatrix,DenseMatrix,DenseMatrixCM>(
		const DistributedMatrix<SparseMatrix>& v,
		const DistributedMatrix<DenseMatrix>& w, const DistributedMatrix<DenseMatrixCM>& h,
		boost::numeric::ublas::matrix<double>& result, const std::string& taskId,
		int tasksPerRank=1, bool asyncRecv = true, int pollDelay = -1);

template void runFunctionPerAssignedBlock3<SparseMatrix, DenseMatrix, DenseMatrixCM, double>(
		mpi2::Channel ch, boost::function<double (const SparseMatrix&, const DenseMatrix&, const DenseMatrixCM&)> f);

}

