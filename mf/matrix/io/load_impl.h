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
#include <mf/matrix/io/load.h>   // compiler hint

namespace mf {

namespace detail {

template<typename M>
struct BlockAndLoadMatrixTask {
	static const std::string id() {return std::string("__mf/matrix/detail/BlockAndLoadMatrixTask") + mpi2::TypeTraits<M>::name(); }
	static void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::string fname, name;
		boost::numeric::ublas::matrix<int> blockLocations;
		std::vector<mf_size_type> blockOffsets1, blockOffsets2;
		MatrixFileFormat format;

		// get data
		ch.recv(name);
		ch.recv(blockLocations);
		ch.recv(blockOffsets1);
		ch.recv(blockOffsets2);
		ch.recv(fname);
		ch.recv(format);

		// create the list of blocks for this node
		int rank = ch.world().rank();
		mf_size_type blocks1 = blockLocations.size1();
		mf_size_type blocks2 = blockLocations.size2();
		std::vector<std::pair<mf_size_type, mf_size_type> > blockList;
		for (mf_size_type b1=0; b1<blocks1; b1++)
			for (mf_size_type b2=0; b2<blocks2; b2++)
				if (blockLocations(b1,b2)==rank)
					blockList.push_back(std::pair<mf_size_type, mf_size_type>(b1,b2));

		// read and block the matrix
		mf_size_type size1, size2;
		std::vector<M*> blocks;
		readMatrixBlocks(fname,
				blocks1, blocks2, blockList, blockOffsets1, blockOffsets2,
				size1, size2, blocks, format);

		// store the blocks in the local environment
		for (unsigned i=0; i<blockList.size(); i++) {
			mf_size_type b1 = blockList[i].first;
			mf_size_type b2 = blockList[i].second;
			mpi2::env().create(defaultBlockName(name, b1, b2), blocks[i]);
		}

		// acknowledge and, send results (first task only)
		ch.send(); // ack
		if (info.groupId() == 0) {
			ch.send(size1);
			ch.send(size2);
		}
	}
};

} // namespace detail

template<typename M>
DistributedMatrix<M> loadMatrix(
		const std::string& name, const boost::numeric::ublas::matrix<int>& blockLocations,
		const std::vector<mf_size_type>& blockOffsets1, // can be empty
		const std::vector<mf_size_type>& blockOffsets2, // can be empty
		const std::string& fname, MatrixFileFormat format = AUTOMATIC) {
	if (blockLocations.size1() > 1 || blockLocations.size2() > 1) {
		LOG4CXX_INFO(detail::logger, "File '" << fname << "' is not blocked; it will be "
				"blocked automatically");
	}

	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	const unsigned m = world.size();

	std::vector<mpi2::Channel> channels(m, mpi2::UNINITIALIZED);
	tm.spawnAll<detail::BlockAndLoadMatrixTask<M> >(channels);

	mpi2::sendAll(channels, name);
	mpi2::sendAll(channels, blockLocations);
	mpi2::sendAll(channels, blockOffsets1);
	mpi2::sendAll(channels, blockOffsets2);
	mpi2::sendAll(channels, fname);
	mpi2::sendAll(channels, format);
	mpi2::recvAll(channels);

	mf_size_type size1, size2;
	channels[0].recv(size1);
	channels[0].recv(size2);
	DistributedMatrix<M> result(name, size1, size2, blockLocations);
	return result;
}

template<typename M>
DistributedMatrix<M> loadMatrix(
		const std::string& name, mf_size_type blocks1, mf_size_type blocks2,
		bool partitionByRow,
		const std::string& fname, MatrixFileFormat format = AUTOMATIC) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	boost::numeric::ublas::matrix<int> blockLocations(blocks1,blocks2);
	computeDefaultBlockLocations(world.size(), blocks1, blocks2, partitionByRow, blockLocations);
	std::vector<mf_size_type> emptyOffsets;
	return loadMatrix<M>(name, blockLocations, emptyOffsets, emptyOffsets,
			fname, format);
}

//template<typename M>
//DistributedMatrix<M> loadMatrix(
//		const std::string& name, std::vector<mf_size_type> blockOffsets1, std::vector<mf_size_type> blockOffsets2,
//		bool partitionByRow,
//		const std::string& fname, MatrixFileFormat format = AUTOMATIC) {
//	if (mf::detail::endsWith(fname, ".xml")){
//		std::cerr<<"You gave me Offsets! I need to read an mmc or mma file, not an xml";
//	}
//	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
//	boost::mpi::communicator& world = tm.world();
//	mf_size_type blocks1=blockOffsets1.size();
//	mf_size_type blocks2=blockOffsets2.size();
//	boost::numeric::ublas::matrix<int> blockLocations(blocks1,blocks2);
//	computeDefaultBlockLocations(world.size(), blocks1, blocks2, partitionByRow, blockLocations);
//	std::vector<mf_size_type> emptyOffsets;
//	return loadMatrix<M>(name, blockLocations, blockOffsets1, blockOffsets2,
//			fname, format);
//}

template<typename M>
DistributedMatrix<M> distributeMatrix(
		const std::string& name, mf_size_type blocks1, mf_size_type blocks2, bool partitionByRow,
		const M& m) {
	LOG4CXX_WARN(detail::logger, "distributeMatrix() is currently very inefficient; "
			"try to use loadMatrix() instead");

	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	boost::numeric::ublas::matrix<int> blockLocations(blocks1,blocks2);
	computeDefaultBlockLocations(world.size(), blocks1, blocks2, partitionByRow, blockLocations);
	DistributedMatrix<M> result = DistributedMatrix<M>(name, m.size1(), m.size2(), blockLocations);
	M temp;
	for (mf_size_type b1=0; b1<blocks1; b1++) {
		for (mf_size_type b2=0; b2<blocks2; b2++) {
			projectSubrange(m, temp,
					result.blockOffsets1()[b1], b1+1<blocks1 ? result.blockOffsets1()[b1+1] : m.size1(),
					result.blockOffsets2()[b2], b2+1<blocks2 ? result.blockOffsets2()[b2+1] : m.size2());
			result.block(b1,b2).createCopy(temp);
		}
	}
	return result;
}

} // namespace mf
